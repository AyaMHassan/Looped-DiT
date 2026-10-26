"""Giles diagnostics of a trained checkpoint, without training.

    python -m looped_dit.diagnose --checkpoint CKPT --repeats 3 --replicates 4

Loads the checkpoint, draws real (image, caption) batches through the training data loader (the gradient variance
needs data: point `--set chunk_dirs=[...]` or `--set webdataset_sources=...` at data that exists where this runs) and
runs the same measurement as the periodic diagnostics of train.py: the per-sample gradient variance V_k and the
compute cost C_k of every level of the telescope and of the baseline objective, the Giles root, the ideal speedup
and the equal-cost allocation. The measurement is repeated and averaged, because a single one is noisy.

    --base-depth 2      telescope over depths 2..N (the base level runs 2 loops and reads the shallower exits)
    --allocation 17,9,4,2   time a real packed MLMC step (one backward) at this allocation, shallowest level first, and give its
                        time x variance efficiency against the baseline step (repeatable; --allocation-budgets adds the
                        equal-compute allocations); below 1 = better than the baseline per unit of time. Each of those budgets B
                        also gets the target-variance search: the cheapest allocation that reaches the baseline's variance
                        V_ref / B (N_k ~ sqrt(V_k / C_k)), timed as real packed steps, with its speedup against B baseline samples
    --ref-sweep 32 64 96 128   time the baseline at each micro-batch size (does it fit, ms per sample, peak memory) and give the
                        allocations' efficiency against the baseline at each size that fit
    --objective final   the final exit only (all weights 1, plain depth-N reference); auto follows the checkpoint's
                        deep_supervision setting
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch
import yaml

from .config import TrainConfig
from .diffusion import deep_supervision_weights, default_level_allocation, final_exit_weights, mlmc_level_weights
from .mlmc_diagnostics import (
    baseline_gradient_estimate,
    budget_sweep,
    build_report,
    compare_estimators,
    equal_cost_sizes,
    estimator_statistics,
    level_block_costs,
    make_losses,
    measure,
    reference_block_cost,
    allocation_efficiency,
    SampleStream,
    target_allocation,
    telescope_gradient_estimate,
    tile_batch,
    peak_memory_gb,
    reset_peak_memory,
    time_packed_steps,
    time_per_sample_fit,
    timing_sizes_from_variances,
)
from .model import LoopedMMDiT


def load_for_diagnostics(
    checkpoint: str | Path, device: torch.device, weights: str = "ema", overrides: dict | None = None
) -> tuple[LoopedMMDiT, TrainConfig]:
    """The checkpoint's model in fp32 with trainable parameters (the diagnostics take gradients), and its config."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = TrainConfig.from_dict({**ckpt["config"], **(overrides or {})})
    key = weights if weights in ckpt else "ema"  # released checkpoints may hold only the EMA weights
    if key != weights:
        print(f"checkpoint has no '{weights}' weights; using '{key}'", flush=True)
    model = LoopedMMDiT(**cfg.model_kwargs())
    model.load_state_dict(ckpt[key])
    return model.to(device).train(), cfg


def diagnostic_weights(cfg: TrainConfig, objective: str) -> tuple[list[float], list[float], bool]:
    """(exit weights, level weights, final_only) of the objective the diagnostics evaluate."""
    final = objective == "final" or (objective == "auto" and not cfg.deep_supervision)
    if final:
        return (*final_exit_weights(cfg.num_loops), True)
    return (
        deep_supervision_weights(cfg.num_loops, cfg.deep_supervision_weighting),
        mlmc_level_weights(cfg.num_loops, cfg.deep_supervision_weighting),
        False,
    )


def aggregate_reports(reports: list[dict[str, float]]) -> tuple[dict, dict, dict]:
    """Per-metric mean, standard deviation and median over the repeats, ignoring non-finite values (a metric such as fit/beta
    is NaN when a telescope has too few levels; statistics.stdev would raise on it). NaN where no repeat has a finite value."""
    mean, std, median = {}, {}, {}
    for key in reports[0]:
        values = [r[key] for r in reports if math.isfinite(r[key])]
        mean[key] = statistics.fmean(values) if values else float("nan")
        median[key] = statistics.median(values) if values else float("nan")
        std[key] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else float("nan"))
    return mean, std, median


def run_diagnostics(
    model: LoopedMMDiT,
    cfg: TrainConfig,
    next_batch: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    autocast: Callable,
    base_depth: int = 1,
    objective: str = "auto",
    replicates: int = 8,
    repeats: int = 3,
    alloc_batch: int | None = None,
    ref_batch: int | None = None,
    timing_reps: int = 3,
    budgets: list[int] | None = None,
    max_forward_batch: int = 128,
    sweep_iterations: int = 3,
    common_noise: bool = True,
    validate_budgets: list[int] | None = None,
    validate_estimates: int = 16,
    allocations: list[list[int]] | None = None,
    allocation_budgets: list[int] | None = None,
    ref_sweep: list[int] | None = None,
) -> dict:
    """Run the measurement `repeats` times on fresh replicate batches (costs are timed once and reused) and return
    the per-metric mean and standard deviation over the repeats, with the per-repeat reports."""
    num_levels = cfg.num_loops
    if num_levels < 2:
        raise ValueError("the telescope needs at least two loops")
    base_depth = max(1, min(base_depth, num_levels - 1))
    depths = list(range(base_depth, num_levels + 1))
    alloc_batch = alloc_batch or cfg.micro_batch_size
    ref_batch = ref_batch or cfg.micro_batch_size
    exit_weights, level_weights, final_only = diagnostic_weights(cfg, objective)
    pre, core, post = (int(n) for n in cfg.loop_split)
    block_costs = level_block_costs(pre, core, post, num_levels, base_depth, exit_weights)
    block_ref = reference_block_cost(pre, core, post, num_levels, exit_weights)
    # a final-exit-only telescope is also measured against the deep-supervised baseline objective (mlmc_diag/vs_deep/...)
    deep_weights = deep_supervision_weights(num_levels, cfg.deep_supervision_weighting) if final_only else None
    deep_block_ref = reference_block_cost(pre, core, post, num_levels, deep_weights) if final_only else None
    loss_kwargs = dict(
        noise_scale=cfg.noise_scale, t_logit_mean=cfg.t_logit_mean, t_logit_std=cfg.t_logit_std,
        label_drop_rate=cfg.label_drop_rate,
    )
    sizes = lambda V: timing_sizes_from_variances(V, block_costs, alloc_batch, len(depths), block_ref)  # noqa: E731
    cache, reports, first_batch = None, [], None
    for i in range(repeats):
        batches = [next_batch() for _ in range(replicates)]
        first_batch = first_batch or batches[0]
        meas = measure(model, batches, level_weights, exit_weights, loss_kwargs, device, autocast, sizes, ref_batch,
                       timing_reps, cost_cache=cache, base_depth=base_depth, common_noise=common_noise,
                       deep_exit_weights=deep_weights)
        if cache is None:  # the first measurement times the costs: say if the GPU forced smaller batches than asked for
            wanted = sizes(meas.V)
            for depth, asked, used in zip(depths, wanted, meas.timing_sizes):
                if used < asked:
                    print(f"WARNING: level {depth} was timed at {used:.0f} samples, not {asked} (out of memory, batch halved)", flush=True)
            if meas.ref_batch < ref_batch:
                print(f"WARNING: the reference was timed at {meas.ref_batch:.0f} samples, not {ref_batch} (out of memory, batch halved): "
                      "every cost ratio is relative to a reference with more per-call overhead", flush=True)
            cache = meas.cost_cache()
        report = build_report(meas, block_costs, block_ref, depths=depths, budget_batch=alloc_batch,
                              deep_block_ref=deep_block_ref)
        report["mlmc_diag/base_depth"], report["mlmc_diag/final_exit_only"] = float(base_depth), float(final_only)
        reports.append(report)
        print(f"repeat {i + 1}/{repeats}: giles_root {report['mlmc_diag/giles_root']:.3f}, "
              f"blocks root {report['mlmc_diag/blocks/giles_root']:.3f}", flush=True)
    mean, std, median = aggregate_reports(reports)
    sweep = None
    if budgets:  # what a bigger compute budget buys, with the variances averaged over the repeats
        level_loss, _ = make_losses(model, level_weights, exit_weights, loss_kwargs, autocast, base_depth)
        params = [p for p in model.parameters() if p.requires_grad]

        def time_level(j: int, n: int) -> tuple[float, int]:
            return time_per_sample_fit(lambda m: level_loss(depths[j], tile_batch(first_batch, m)), n, params, device, timing_reps)

        sweep = budget_sweep(
            [mean[f"mlmc_diag/level{d}/grad_variance"] for d in depths], mean["mlmc_diag/reference/grad_variance"],
            mean["mlmc_diag/reference/cost_ms_per_sample"], block_costs, block_ref, time_level, budgets, max_forward_batch,
            sweep_iterations)
        print("budget sweep done", flush=True)
    # a final-exit-only telescope is also compared with the deep-supervised reference (see above)
    references = [("final" if final_only else "baseline", "mlmc_diag/reference")]
    if final_only:
        references.append(("deep", "mlmc_diag/vs_deep/reference"))
    ref_rows = None
    if ref_sweep:  # the baseline's own cost at larger micro-batches: does it fit, and is it cheaper per sample there?
        params = [p for p in model.parameters() if p.requires_grad]
        made = make_losses(model, level_weights, exit_weights, loss_kwargs, autocast, base_depth, deep_weights)
        ref_fns = [made[1]] + ([made[2]] if final_only else [])
        ref_rows = []
        for (name, prefix), fn in zip(references, ref_fns):
            for size in ref_sweep:
                reset_peak_memory(device)
                ms, used = time_per_sample_fit(lambda n: fn(tile_batch(first_batch, n)), size, params, device, timing_reps)
                ref_rows.append({"reference": name, "prefix": prefix, "requested": size, "used": used, "ms_per_sample": ms,
                                 "step_ms": ms * used, "peak_gb": peak_memory_gb(device)})
        print("reference sweep done", flush=True)
    table, result_target = None, None
    if allocations or allocation_budgets:  # real packed steps timed per allocation: time x variance against the baseline step
        params = [p for p in model.parameters() if p.requires_grad]
        V = [mean[f"mlmc_diag/level{d}/grad_variance"] for d in depths]
        candidates = [list(a) for a in (allocations or [])]
        for budget in allocation_budgets or []:  # the equal-compute allocation (block costs) of each baseline batch
            candidates.append(equal_cost_sizes(V, block_costs, budget * block_ref) or default_level_allocation(max(budget, len(V)), len(V)))
        candidates = [c for i, c in enumerate(candidates) if c not in candidates[:i]]
        for c in candidates:
            if len(c) != len(depths):
                raise ValueError(f"allocation {c} needs {len(depths)} entries, one per level (depths {depths})")
        memory: list[float] = []
        times = time_packed_steps(model, level_weights, exit_weights, loss_kwargs, autocast, base_depth, first_batch, candidates,
                                  params, device, timing_reps, memory)
        m_ref = mean["mlmc_diag/reference/timing_batch"]
        table = []
        for sizes, ms, peak in zip(candidates, times, memory):
            row = {"sizes": sizes, "step_ms": ms, "peak_gb": peak, "samples": sum(sizes),
                   "variance": sum(v / n for v, n in zip(V, sizes))}
            for name, prefix in references:
                ref_ms = mean[f"{prefix}/cost_ms_per_sample"] * m_ref
                row[f"efficiency_vs_{name}"] = allocation_efficiency(V, mean[f"{prefix}/grad_variance"], ms, sizes, ref_ms, m_ref)
            for r in ref_rows or []:  # against the baseline at every micro-batch of the sweep that ran as requested
                if r["used"] == r["requested"]:
                    row[f"efficiency_vs_{r['reference']}@{r['used']}"] = allocation_efficiency(
                        V, mean[f"{r['prefix']}/grad_variance"], ms, sizes, r["step_ms"], r["used"])
            table.append(row)
        print("packed-step timing done", flush=True)
        if allocation_budgets:  # the cheapest allocation that reaches the baseline's variance at batch B, timed for real
            target_table = []
            costs = {"measured": [mean[f"mlmc_diag/level{d}/cost_ms_per_sample"] for d in depths], "blocks": block_costs}
            cap = max_forward_batch
            for name, prefix in references:
                for budget in allocation_budgets:
                    target = mean[f"{prefix}/grad_variance"] / budget
                    base_ms = mean[f"{prefix}/cost_ms_per_sample"] * budget  # B/m micro-steps of m samples at ms/sample
                    for model_name, C in costs.items():
                        plan = target_allocation(V, C, target, cap)
                        step_ms = float("nan")
                        while plan is not None:
                            a, micro, total = plan
                            step_ms = time_packed_steps(model, level_weights, exit_weights, loss_kwargs, autocast, base_depth,
                                                        first_batch, [micro], params, device, timing_reps)[0]
                            if math.isfinite(step_ms) or sum(micro) <= len(micro) or a >= 1024:
                                break  # timed, or cannot be split smaller
                            plan = target_allocation(V, C, target, cap, micro_batches=a * 2)
                        if plan is None:
                            continue
                        total_ms = a * step_ms
                        target_table.append({
                            "reference": name, "budget": budget, "cost_model": model_name, "micro_batches": a, "sizes": micro,
                            "total_sizes": total, "variance_over_target": sum(v / n for v, n in zip(V, total)) / target,
                            "step_ms": step_ms, "mlmc_ms": total_ms, "baseline_ms": base_ms, "speedup": base_ms / total_ms,
                        })
            result_target = target_table
            print("target-variance allocation timing done", flush=True)
    validation = None
    if validate_budgets:  # run both gradient estimators for real and compare them
        level_loss, ref_loss = make_losses(model, level_weights, exit_weights, loss_kwargs, autocast, base_depth)
        params = [p for p in model.parameters() if p.requires_grad]
        V = [mean[f"mlmc_diag/level{d}/grad_variance"] for d in depths]
        V_ref = mean["mlmc_diag/reference/grad_variance"]
        stream = SampleStream(next_batch)

        validation = []
        for budget in validate_budgets:
            # The equal-compute allocation: the baseline's compute for `budget` samples (in block counts) spent optimally over
            # the levels, each level run in forward chunks of at most max_forward_batch.
            samples = equal_cost_sizes(V, block_costs, budget * block_ref) or default_level_allocation(max(budget, len(V)), len(V))
            forward = [min(n, max_forward_batch) for n in samples]
            print(f"validating budget {budget}: equal-compute telescope samples {samples}, forward batches {forward}", flush=True)
            base_stats = estimator_statistics(
                lambda: baseline_gradient_estimate(ref_loss, stream, params, budget, ref_batch), validate_estimates, device)
            tele_stats = estimator_statistics(
                lambda: telescope_gradient_estimate(level_loss, depths, stream, params, samples, forward), validate_estimates, device)
            result_row = compare_estimators(base_stats, tele_stats, V_ref / budget, sum(v / n for v, n in zip(V, samples)))
            result_row.update(budget=float(budget), samples=samples, forward_batch=forward, num_estimates=validate_estimates)
            validation.append(result_row)
            del base_stats, tele_stats
        print("validation done", flush=True)
    return {"mean": mean, "std": std, "median": median, "reports": reports, "budget_sweep": sweep, "validation": validation,
            "allocation_table": table, "target_table": result_target, "ref_sweep": ref_rows,
            "max_forward_batch": max_forward_batch, "depths": depths, "base_depth": base_depth,
            "final_exit_only": final_only, "replicates": replicates, "repeats": repeats,
            "alloc_batch": alloc_batch, "ref_batch": ref_batch}


def print_summary(result: dict) -> None:
    m, s, med = result["mean"], result["std"], result["median"]
    print(f"\nTelescope over depths {result['depths']}, objective: "
          f"{'final exit only' if result['final_exit_only'] else 'deep supervision'}; "
          f"{result['repeats']} repeats of {result['replicates']} replicates (mean +- std)")
    for tag, label in (("", "measured costs"), ("blocks/", "block costs   ")):
        root, speed = m[f"mlmc_diag/{tag}giles_root"], m[f"mlmc_diag/{tag}ideal_speedup"]
        print(f"  {label}: giles_root {root:.3f} +- {s[f'mlmc_diag/{tag}giles_root']:.3f} (median {med[f'mlmc_diag/{tag}giles_root']:.3f})"
              f"   ideal speedup {speed:.3f}")
    print(f"  reference timed at {m['mlmc_diag/reference/timing_batch']:.0f} samples")
    if "mlmc_diag/equal_cost/variance_gain" in m:
        print(f"  equal-cost allocation of {result['alloc_batch']} baseline samples: {m['mlmc_diag/equal_cost/total_samples']:.0f} samples, "
              f"cost vs baseline {m['mlmc_diag/equal_cost/cost_vs_baseline']:.3f}, "
              f"variance gain {m['mlmc_diag/equal_cost/variance_gain']:.3f} (blocks {m['mlmc_diag/blocks/equal_cost/variance_gain']:.3f})")
    print(f"\n  {'depth':>5} {'var ratio':>10} {'cost ratio':>11} {'(blocks)':>9} {'contribution':>13} {'(blocks)':>9} "
          f"{'optimal frac':>13} {'timed at n':>11}")
    for d in result["depths"]:
        p = f"mlmc_diag/level{d}"
        print(f"  {d:>5} {m[f'{p}/variance_ratio']:>10.4f} {m[f'{p}/cost_ratio']:>11.3f} {m[f'mlmc_diag/blocks/level{d}/cost_ratio']:>9.3f} "
              f"{m[f'{p}/giles_contribution']:>13.3f} {m[f'mlmc_diag/blocks/level{d}/giles_contribution']:>9.3f} "
              f"{m[f'{p}/optimal_sample_fraction']:>13.3f} {m[f'{p}/timing_batch']:>11.1f}")


def print_sweep(result: dict) -> None:
    rows = result["budget_sweep"]
    if not rows:
        return
    ref = result["ref_batch"]
    print(f"\nBudget sweep: for a baseline batch B (in samples) the equal-cost allocation of the telescope. The baseline runs "
          f"micro-batches of {ref} (B/{ref} forward calls per step); each level runs micro-batches of up to "
          f"{result['max_forward_batch']} samples. The allocation is re-optimized with the measured costs, so 'cost vs base' "
          f"should be near 1 (above 1 at a small budget: every level needs at least one sample).")
    print(f"  {'B':>5} {'base calls':>10} {'samples':>8}  {'per level':<26} {'micro-batch':<20} {'calls':<14} {'cost vs base':>12} "
          f"{'root':>6} {'ideal speedup':>14} {'variance gain':>14} {'(blocks)':>9}")
    for r in rows:
        print(f"  {r['budget']:>5.0f} {math.ceil(r['budget'] / ref):>10} {r['total_samples']:>8.0f}  "
              f"{str([int(x) for x in r['samples']]):<26} {str([int(x) for x in r['forward_batch']]):<20} "
              f"{str([int(x) for x in r['forward_calls']]):<14} {r['cost_vs_baseline']:>12.3f} {r['root']:>6.3f} "
              f"{r['ideal_speedup']:>14.3f} {r['variance_gain']:>14.3f} {r['blocks_variance_gain']:>9.3f}")


def print_ref_sweep(result: dict) -> None:
    rows = result.get("ref_sweep")
    if not rows:
        return
    print("\nBaseline micro-batch sweep: the reference loss (forward + backward) timed at several micro-batch sizes. 'ran at' below "
          "'asked' means the GPU ran out of memory and the batch was halved: that size does not fit. A falling ms/sample means the "
          "baseline is still gaining from a larger micro-batch (and the MLMC speedup against it shrinks). The diagnostic holds no "
          "optimizer state or EMA, so a size that fits here can still run out of memory in training.")
    print(f"  {'vs':<9} {'asked':>6} {'ran at':>7} {'ms/sample':>10} {'step ms':>9} {'peak GB':>8}")
    for r in rows:
        print(f"  {r['reference']:<9} {r['requested']:>6} {r['used']:>7} {r['ms_per_sample']:>10.2f} {r['step_ms']:>9.1f} "
              f"{r['peak_gb']:>8.1f}")


def print_allocation_table(result: dict) -> None:
    rows = result.get("allocation_table")
    if not rows:
        return
    m = result["mean"]
    names = [k.removeprefix("efficiency_vs_") for k in rows[0] if k.startswith("efficiency_vs_")]
    key_name = "deep" if "deep" in names else names[0]
    print(f"\nPacked-step timing: one real MLMC step per allocation (all levels in one micro-batch, one summed loss, one backward) "
          f"against a baseline micro-batch of {m['mlmc_diag/reference/timing_batch']:.0f} samples. efficiency = (step time x "
          f"gradient variance) / (the baseline's); below 1 the telescope is better per unit of time and 1/efficiency is its speedup. "
          f"The ratio does not change with gradient accumulation. peak GB is the GPU memory of the step (no optimizer state).")
    print(f"  {'samples per level':<26} {'total':>6} {'step ms':>9} {'peak GB':>8} " + " ".join(f"{'eff vs ' + n:>16}" for n in names))

    def order(r):
        v = r.get(f"efficiency_vs_{key_name}")
        return v if v is not None and math.isfinite(v) else float("inf")

    for r in sorted(rows, key=order):
        print(f"  {str(r['sizes']):<26} {r['samples']:>6} {r['step_ms']:>9.1f} {r['peak_gb']:>8.1f} "
              + " ".join(f"{r[f'efficiency_vs_{n}']:>16.3f}" for n in names))
    print("  (final = against the plain final-loss step; deep = against the deep-supervised step, the baseline you would train otherwise;"
          " name@N = against the baseline at micro-batch N)" if any("@" in n for n in names) or len(names) > 1 else "")


def print_target_table(result: dict) -> None:
    rows = result.get("target_table")
    if not rows:
        return
    print("\nTarget-variance allocation: for each baseline batch B the cheapest telescope that reaches the baseline's gradient variance "
          "V_ref / B (N_k ~ sqrt(V_k / C_k), rounded up, split over equal micro-batches), timed as real packed steps. speedup = "
          "baseline time for B samples / telescope time (above 1: MLMC reaches the same noise faster). 'cost model' is the cost "
          "(measured ms or blocks) used to choose the allocation; take the fastest row per B.")
    print(f"  {'vs':<9} {'B':>5} {'cost model':<10} {'micro':>5} {'samples per level / micro-batch':<32} {'var/target':>10} "
          f"{'mlmc ms':>9} {'base ms':>9} {'speedup':>8}")
    for r in sorted(rows, key=lambda r: (r["reference"], r["budget"], -(r["speedup"] if math.isfinite(r["speedup"]) else 0))):
        print(f"  {r['reference']:<9} {r['budget']:>5} {r['cost_model']:<10} {r['micro_batches']:>5} {str(r['sizes']):<32} "
              f"{r['variance_over_target']:>10.3f} {r['mlmc_ms']:>9.1f} {r['baseline_ms']:>9.1f} {r['speedup']:>8.2f}")


def print_validation(result: dict) -> None:
    rows = result.get("validation")
    if not rows:
        return
    print(f"\nEmpirical test: both gradient estimators run {rows[0]['num_estimates']} times on fresh data at each budget B (baseline: B samples in "
          f"micro-batches of {result['ref_batch']}; telescope: the equal-compute allocation, i.e. the same block-count budget). Variances are of the gradient estimates themselves.")
    print(f"  {'B':>5} {'telescope samples':<24} {'time base/tele (s)':>20} {'time ratio':>10} {'var base/var tele':>18} {'(predicted)':>12} "
          f"{'(even, odd)':>14} {'efficiency gain':>16} {'bias stat':>10}")
    for r in rows:
        print(f"  {r['budget']:>5.0f} {str(r['samples']):<24} {r['time_baseline_s']:>9.2f}/{r['time_telescope_s']:<10.2f} {r['time_ratio']:>10.2f} "
              f"{r['variance_ratio']:>18.2f} {r['predicted_variance_ratio']:>12.2f} "
              f"{r['variance_ratio_even']:>7.2f},{r['variance_ratio_odd']:<6.2f} {r['efficiency_gain']:>16.2f} {r['bias_statistic']:>10.2f}")
    print("  time ratio shows how the equal compute translates into wall time (above 1: the telescope is slower, from "
          "small-batch overhead). efficiency gain = (variance x time of the baseline) / (variance x time of the telescope): above 1 means "
          "the telescope gives a less noisy gradient for the same time. bias stat near 1 means the two estimators agree on the mean gradient.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", default="ema", choices=["ema", "model"])
    parser.add_argument("--base-depth", type=int, default=1, help="depth of the telescope's base level (1 = all levels)")
    parser.add_argument("--objective", default="auto", choices=["auto", "deep_supervision", "final"])
    parser.add_argument("--replicates", type=int, default=8, help="micro-batches per measurement (>= 2)")
    parser.add_argument("--no-common-noise", action="store_true", help="draw t / noise / dropout independently for every term (noisier ratios)")
    parser.add_argument("--repeats", type=int, default=3, help="measurements to average")
    parser.add_argument("--batch-size", type=int, help="samples per replicate (default: the config's micro_batch_size)")
    parser.add_argument("--alloc-batch", type=int, help="baseline batch whose compute the allocation spends (default micro_batch_size)")
    parser.add_argument("--ref-batch", type=int, help="batch the reference is timed at (default micro_batch_size)")
    parser.add_argument("--timing-reps", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", metavar="B",
                        help="budget sweep: baseline batch sizes (samples) whose compute is spent on the telescope, e.g. 32 64 128 256")
    parser.add_argument("--max-forward-batch", type=int, default=128, help="largest forward batch a level runs at in the sweep")
    parser.add_argument("--sweep-iters", type=int, default=3, help="iterations of the cost/allocation fixed point per budget")
    parser.add_argument("--validate", type=int, nargs="+", metavar="B",
                        help="empirical test: run the baseline and the equal-time telescope gradient estimators for these baseline "
                             "budgets (samples) and compare their variance and time")
    parser.add_argument("--validate-estimates", type=int, default=16, help="gradient estimates per estimator and budget")
    parser.add_argument("--allocation", action="append", metavar="N1,N2,..", help="time a real packed MLMC step at this samples-per-level "
                        "allocation (shallowest level first, one entry per level; repeat the flag for several) and report its time x variance "
                        "efficiency against the baseline")
    parser.add_argument("--allocation-budgets", type=int, nargs="+", metavar="B",
                        help="also time the equal-compute allocation of each baseline batch B (block costs)")
    parser.add_argument("--ref-sweep", type=int, nargs="+", metavar="N",
                        help="time the baseline loss at each of these micro-batch sizes (does it fit, ms per sample, peak memory) and "
                             "give every --allocation's efficiency against the baseline at each size that fit, e.g. 32 64 96 128")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-encoder", help="override the text encoder path from the checkpoint")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values, e.g. data paths")
    parser.add_argument("--out", default="giles_diagnostics.json")
    args = parser.parse_args(argv)
    if args.replicates < 2:
        parser.error("--replicates must be >= 2 (a variance needs two replicates)")

    from .data import make_loader
    from .pipeline import TextEncoder

    overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)}
    if args.batch_size:
        overrides["micro_batch_size"] = args.batch_size
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.bfloat16 if cuda else torch.float32
    torch.manual_seed(args.seed)
    model, cfg = load_for_diagnostics(args.checkpoint, device, args.weights, overrides)
    text_encoder = TextEncoder(args.text_encoder or cfg.text_encoder, cfg.prompt_length, device)
    batches = iter(make_loader(cfg, seed_offset=args.seed))

    def next_batch():
        batch = next(batches)
        images = batch["pixel_values"].to(device).float().mul_(1 / 127.5).add_(-1.0)
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=cuda):
            text = text_encoder.encode(ids, mask).to(dtype)
        return images, text, mask

    autocast = lambda: torch.autocast("cuda", dtype=dtype, enabled=cuda)  # noqa: E731
    result = run_diagnostics(model, cfg, next_batch, device, autocast, args.base_depth, args.objective, args.replicates,
                             args.repeats, args.alloc_batch, args.ref_batch, args.timing_reps,
                             args.budgets, args.max_forward_batch, args.sweep_iters, not args.no_common_noise,
                             args.validate, args.validate_estimates,
                             [[int(x) for x in a.split(",")] for a in args.allocation or []], args.allocation_budgets,
                             args.ref_sweep)
    print_summary(result)
    print_sweep(result)
    print_ref_sweep(result)
    print_allocation_table(result)
    print_target_table(result)
    print_validation(result)
    result["checkpoint"] = str(args.checkpoint)
    result["reports"] = [{k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()}
                         for r in result["reports"]]
    for row in (result["budget_sweep"] or []) + (result["validation"] or []) + (result["allocation_table"] or []) + (result["target_table"] or []) + (result["ref_sweep"] or []):
        for k, v in row.items():
            if isinstance(v, float) and not math.isfinite(v):
                row[k] = None
    for key in ("mean", "std", "median"):
        result[key] = {k: (None if not math.isfinite(v) else v) for k, v in result[key].items()}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
