"""Giles diagnostics for the MLMC loss over loop depth.

Level 1 is the depth-1 exit; level k >= 2 is the correction F_k - S_{k-1} from one pass to
depth k (exits k and k-1 on the same input). The estimator of the baseline objective
J = sum_r w_r F_r is sum_k W_k term_k with independent samples per level.

For a level with per-sample gradient variance V_k and cost C_k, and a reference estimator
(the baseline objective on full-depth passes) with V_ref and C_ref, Giles' optimal allocation
N_k ~ sqrt(V_k / C_k) gives

    R = sum_k sqrt( (V_k / V_ref) * (C_k / C_ref) ),    ideal speedup = 1 / R^2.

V_k is the variance of the *parameter gradient* (trace of its covariance), not of the scalar
loss: loss agreement between two exits does not imply gradient agreement. Costs are measured at
the batch sizes that really run (forward + autograd.grad, ms per sample) and also counted in transformer blocks. The scalar
loss statistics are logged separately as proxies and never enter the formula.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.distributed as dist
from torch import nn

from .diffusion import (
    allocate_by_fractions, default_level_allocation, MLMCObjective, mlmc_base_term, mlmc_level_term, split_evenly, training_loss,
)

EPS = 1e-12


# ---------------------------------------------------------------------------
# Block-count cost model
# ---------------------------------------------------------------------------


def level_block_costs(
    pre: int, core: int, post: int, num_levels: int, base_depth: int = 1, exit_weights: list[float] | None = None
) -> list[float]:
    """Blocks per sample of the levels of a telescope over depths base_depth..num_levels (num_levels is the top
    depth K). The base level runs base_depth loops and decodes its own exit plus every shallower exit that has a
    weight (all of them with deep supervision, none without); level k > base_depth runs k loops and decodes two exits
    (k and k-1) from the shared pass."""
    below = base_depth - 1 if exit_weights is None else sum(1 for w in exit_weights[: base_depth - 1] if w != 0)
    return [float(pre + base_depth * core + (1 + below) * post)] + [
        float(pre + k * core + 2 * post) for k in range(base_depth + 1, num_levels + 1)
    ]


def reference_block_cost(pre: int, core: int, post: int, num_levels: int, exit_weights: list[float] | None = None) -> float:
    """Blocks per sample of the baseline objective: K loops, every exit with a weight decoded (all K with deep
    supervision, only the final one without)."""
    exits = num_levels if exit_weights is None else sum(1 for w in exit_weights if w != 0)
    return float(pre + num_levels * core + exits * post)


def baseline_block_cost(pre: int, core: int, post: int, num_loops: int, deep_supervision: bool) -> float:
    return float(pre + num_loops * core + (num_loops if deep_supervision else 1) * post)


# ---------------------------------------------------------------------------
# Giles formulas (pure)
# ---------------------------------------------------------------------------


def giles_root(V: list[float], C: list[float], V_ref: float, C_ref: float) -> float:
    """R = sum_k sqrt((V_k/V_ref)(C_k/C_ref)); NaN when any input is non-positive or non-finite."""
    values = list(V) + list(C) + [V_ref, C_ref]
    if not V or len(V) != len(C) or not all(math.isfinite(x) and x > 0 for x in values):
        return float("nan")
    return sum(math.sqrt((v / V_ref) * (c / C_ref)) for v, c in zip(V, C))


def giles_allocation(V: list[float], C: list[float]) -> dict[str, list[float]]:
    """Optimal sample fractions N_k/sum N ~ sqrt(V_k/C_k) and work shares ~ sqrt(V_k C_k)."""
    if not V or len(V) != len(C) or not all(math.isfinite(x) and x > 0 for x in list(V) + list(C)):
        nan = [float("nan")] * len(V)
        return {"sample_fraction": nan, "work_share": list(nan)}
    ratio = [math.sqrt(v / c) for v, c in zip(V, C)]
    work = [math.sqrt(v * c) for v, c in zip(V, C)]
    return {"sample_fraction": [r / sum(ratio) for r in ratio], "work_share": [w / sum(work) for w in work]}


def realized_gain(V: list[float], C: list[float], n: list[float], V_ref: float, C_ref: float) -> float:
    """Variance reduction at equal cost of an allocation n (samples per level) against the
    reference estimator given the same total cost: (V_ref C_ref / T) / sum_k (V_k / n_k)."""
    cost = sum(nk * ck for nk, ck in zip(n, C))
    var = sum(v / nk for v, nk in zip(V, n))
    if not (cost > 0 and var > 0 and V_ref > 0 and C_ref > 0):
        return float("nan")
    return (V_ref * C_ref / cost) / var


def equal_cost_sizes(V: list[float], costs: list[float], budget: float) -> list[int]:
    """Samples per level that spend `budget` (in the units of `costs`) optimally: N_k = budget * sqrt(V_k / C_k) /
    sum_j sqrt(V_j C_j), at least one per level. The variance of the estimator is then (sum_j sqrt(V_j C_j))^2 / budget,
    and the units of V cancel. Returns an empty list when the inputs are unusable."""
    if not V or len(V) != len(costs) or not all(math.isfinite(x) and x > 0 for x in list(V) + list(costs) + [budget]):
        return []
    root = sum(math.sqrt(v * c) for v, c in zip(V, costs))
    return [max(1, int(budget * math.sqrt(v / c) / root + 0.5)) for v, c in zip(V, costs)]


def timing_sizes_from_variances(
    V: list[float], block_costs: list[float], batch: int, num_levels: int, block_ref: float | None = None
) -> list[int]:
    """Batch sizes the levels are timed at in --baseline mode, i.e. the allocation an --mlmc run would use.

    With block_ref (the reference's block cost) this is the equal-cost allocation: the compute budget is that of
    the baseline's batch, batch * block_ref, spent optimally over the levels (equal_cost_sizes, with block costs, so
    there is no circularity with the timing). Because the levels are cheaper than the reference this uses more than
    `batch` samples in total, a bigger micro-batch for the same compute. Without block_ref the same `batch` samples are
    split in the Giles-optimal proportions instead. If the variances are unusable it falls back to a halving split."""
    if batch < num_levels:
        return [1] * num_levels
    if block_ref is not None:
        sizes = equal_cost_sizes(V, block_costs, batch * block_ref)
        return sizes or default_level_allocation(batch, num_levels)
    fractions = giles_allocation(V, block_costs)["sample_fraction"]
    if any(not math.isfinite(f) for f in fractions):
        return default_level_allocation(batch, num_levels)
    return allocate_by_fractions(batch, fractions)


def log2_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of log2(y) against x over the positive points (NaN with < 2)."""
    pts = [(x, math.log2(y)) for x, y in zip(xs, ys) if y > 0 and math.isfinite(y)]
    if len(pts) < 2:
        return float("nan")
    mx, my = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class Measurements:
    """Per-rank raw measurements; lists are indexed by level 1..K."""

    V: list[float] = field(default_factory=list)  # per-sample gradient variance of each weighted level term
    C_ms: list[float] = field(default_factory=list)  # forward + grad ms per sample
    grad_sq: list[float] = field(default_factory=list)  # mean squared gradient norm of the term
    loss_mean: list[float] = field(default_factory=list)  # proxies: mean / variance of the scalar term
    loss_var: list[float] = field(default_factory=list)
    V_ref: float = float("nan")
    C_ref_ms: float = float("nan")
    grad_sq_ref: float = float("nan")
    samples_per_replicate: int = 0
    timing_sizes: list[float] = field(default_factory=list)  # the batch size each level was timed at (after any out-of-memory halving)
    ref_batch: float = 0.0
    # The deep-supervised baseline objective as a second reference (only measured for a final-exit-only telescope).
    V_deep: float = float("nan")
    C_deep_ms: float = float("nan")
    grad_sq_deep: float = float("nan")
    # The samples per level the costs stand for (the equal-cost allocation, or mlmc_batch), and the batch size each level's
    # timing asked for: its largest chunk when levels run in chunks. timing_sizes below timing_wanted = out-of-memory halving.
    alloc_sizes: list[float] = field(default_factory=list)
    timing_wanted: list[float] = field(default_factory=list)

    def cost_cache(self) -> CostCache:
        return CostCache(list(self.C_ms), self.C_ref_ms, list(self.timing_sizes), self.ref_batch, self.C_deep_ms,
                         list(self.alloc_sizes), list(self.timing_wanted))

    def flat(self) -> list[float]:
        return [*self.V, *self.C_ms, *self.grad_sq, *self.loss_mean, *self.loss_var,
                self.V_ref, self.C_ref_ms, self.grad_sq_ref, *self.timing_sizes, self.ref_batch,
                self.V_deep, self.C_deep_ms, self.grad_sq_deep, *self.alloc_sizes, *self.timing_wanted]

    @classmethod
    def unflat(cls, values: list[float], num_levels: int, samples: int) -> "Measurements":
        k = num_levels
        return cls(values[:k], values[k:2 * k], values[2 * k:3 * k], values[3 * k:4 * k], values[4 * k:5 * k],
                   values[5 * k], values[5 * k + 1], values[5 * k + 2], samples,
                   values[5 * k + 3:6 * k + 3], values[6 * k + 3], values[6 * k + 4], values[6 * k + 5], values[6 * k + 6],
                   alloc_sizes=values[6 * k + 7:7 * k + 7], timing_wanted=values[7 * k + 7:8 * k + 7])


@dataclass
class CostCache:
    """Last measured costs, reused between the (rarer) cost measurements."""

    C_ms: list[float]
    C_ref_ms: float
    timing_sizes: list[float]
    ref_batch: float
    C_deep_ms: float = float("nan")
    alloc_sizes: list[float] = field(default_factory=list)
    timing_wanted: list[float] = field(default_factory=list)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def replicate_gradient_variance(
    params: list[nn.Parameter],
    replicate_losses: list[Callable[[], torch.Tensor]],
    samples: int,
) -> tuple[float, float, list[float]]:
    """Per-sample gradient variance of an estimator from R independent replicates.

    Each replicate returns a scalar loss that is a mean over `samples` samples. With replicate
    gradients g_r, the trace of the covariance of the batch-mean gradient is estimated by
    (sum ||g_r||^2 - ||sum g_r||^2 / R) / (R - 1), and times `samples` it is the per-sample V
    (which does not depend on the batch size). Uses autograd.grad, so .grad of the parameters (and
    any pending accumulation) is untouched. Returns (V, mean ||g||^2, scalar losses).
    """
    total = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    sq_sum, losses = 0.0, []
    for fn in replicate_losses:
        loss = fn()
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        sq = 0.0
        for acc, g in zip(total, grads):
            if g is not None:
                acc += g.float()
                sq += float(g.float().pow(2).sum())
        sq_sum += sq
        losses.append(float(loss.detach()))
    n = len(replicate_losses)
    total_sq = sum(float(t.pow(2).sum()) for t in total)
    v_batch = max((sq_sum - total_sq / n) / max(n - 1, 1), 0.0)
    return v_batch * samples, sq_sum / n, losses


def with_seed(fn: Callable[[], torch.Tensor], seed: int | None, device: torch.device) -> Callable[[], torch.Tensor]:
    """fn run under a fixed RNG seed (the global RNG state is restored afterwards). Terms that run under the same seed
    draw the same timesteps, noise and label dropout, because each starts by drawing them in the same order."""
    if seed is None:
        return fn

    def wrapped() -> torch.Tensor:
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            return fn()

    return wrapped


def replicate_batches(fetch: Callable[[], tuple[torch.Tensor, ...]], count: int, size: int) -> list[tuple[torch.Tensor, ...]]:
    """`count` replicate batches of `size` samples for the variance measurement, cut from whatever fetch() returns.

    An MLMC run fetches packed batches of sum(mlmc_batch) samples (214 for 180 20 9 5), and measuring the reference and every level
    on a whole packed batch would hold hundreds of baseline-equivalent samples at once, so the batch is split into
    chunks of `size` (the baseline micro-batch) and the incomplete remainder is dropped. A fetched batch smaller than `size` is
    used whole, so the replicates are never empty and the loop ends."""
    out: list[tuple[torch.Tensor, ...]] = []
    while len(out) < count:
        batch = fetch()
        take = min(size, batch[0].shape[0])
        for chunk in zip(*(t.split(take) for t in batch)):
            if chunk[0].shape[0] == take:
                out.append(chunk)
                if len(out) == count:
                    break
    return out


def tile_batch(batch: tuple[torch.Tensor, ...], n: int) -> tuple[torch.Tensor, ...]:
    """n samples made by cycling through the batch (compute does not depend on the values)."""
    idx = torch.arange(n, device=batch[0].device) % batch[0].shape[0]
    return tuple(x[idx] for x in batch)


def time_per_sample(
    make_loss: Callable[[int], Callable[[], torch.Tensor]],
    n: int,
    params: list[nn.Parameter],
    device: torch.device,
    reps: int,
) -> float:
    """Compute-only cost: ms per sample of forward + autograd.grad at batch size n, after one warm-up.
    GPU cost per sample depends strongly on the batch size, so n must be the size that really runs."""
    fn = make_loss(n)

    def once() -> None:
        torch.autograd.grad(fn(), params, allow_unused=True)

    once()
    _sync(device)
    start = time.perf_counter()
    for _ in range(reps):
        once()
    _sync(device)
    return 1000.0 * (time.perf_counter() - start) / (reps * n)


def time_per_sample_fit(
    make_loss: Callable[[int], Callable[[], torch.Tensor]],
    n: int,
    params: list[nn.Parameter],
    device: torch.device,
    reps: int,
) -> tuple[float, int]:
    """time_per_sample at batch size n, halving n while the GPU runs out of memory.
    Returns (ms per sample, the batch size actually timed)."""
    while True:
        try:
            return time_per_sample(make_loss, n, params, device, reps), n
        except RuntimeError as error:  # torch.cuda.OutOfMemoryError is a RuntimeError in every version
            if "out of memory" not in str(error).lower() or n <= 1:
                raise
            torch.cuda.empty_cache()
            n //= 2


def make_losses(
    model: nn.Module,
    level_weights: list[float],
    exit_weights: list[float],
    loss_kwargs: dict,
    autocast: Callable,
    base_depth: int = 1,
    deep_exit_weights: list[float] | None = None,
) -> tuple[Callable, Callable] | tuple[Callable, Callable, Callable]:
    """(level_loss(depth, batch), ref_loss(batch)): functions returning the closure that computes the weighted loss of one
    level term (the base at base_depth, a correction above it) or of the baseline objective on a batch. With
    deep_exit_weights a third function deep_loss(batch) is returned: the deep-supervised baseline objective."""
    final_only = all(w == 0 for w in exit_weights[:-1])  # no deep supervision: the reference is the plain final loss

    def level_loss(depth: int, batch):
        def fn():
            with autocast():
                if depth == base_depth:
                    return mlmc_base_term(model, depth, exit_weights, level_weights, *batch, **loss_kwargs)
                return mlmc_level_term(model, depth, level_weights[depth - 1], *batch, **loss_kwargs)[0]
        return fn

    def ref_loss(batch):
        def fn():
            with autocast():
                return training_loss(model, *batch, None if final_only else exit_weights, **loss_kwargs)[0]
        return fn

    if deep_exit_weights is None:
        return level_loss, ref_loss

    def deep_loss(batch):
        def fn():
            with autocast():
                return training_loss(model, *batch, deep_exit_weights, **loss_kwargs)[0]
        return fn

    return level_loss, ref_loss, deep_loss


def allocation_efficiency(V: list[float], V_ref: float, step_ms: float, sizes: list[int], ref_step_ms: float,
                          ref_batch: float) -> float:
    """Time-variance efficiency ratio of a telescope step against a baseline step: (T(N) sum_k V_k / N_k) / (T_ref V_ref / m).

    T(N) is the measured time of one packed step with N_k samples at level k (one summed loss, one backward), T_ref the time
    of a baseline step on m = ref_batch samples. Accumulating micro-batches multiplies a time and divides a variance by the same
    factor, so the ratio does not depend on the batch per optimizer step. Below 1: the telescope gives a less noisy gradient
    per unit of time (its speedup is 1 / ratio). NaN for unusable inputs."""
    if not (len(V) == len(sizes) and all(n > 0 for n in sizes) and step_ms > 0 and ref_batch > 0):
        return float("nan")
    var, base = sum(v / n for v, n in zip(V, sizes)), ref_step_ms * V_ref / ref_batch
    return step_ms * var / base if var > 0 and base > 0 else float("nan")


def target_allocation(V: list[float], C: list[float], target_var: float, cap: int, micro_batches: int | None = None
                      ) -> tuple[int, list[int], list[int]] | None:
    """Cheapest allocation that reaches a variance target: N_k = s sqrt(V_k / C_k) with s chosen so that sum_k V_k / N_k =
    target_var (Giles; the units of C cancel), rounded up to whole samples. It is split over `a` equal micro-batches so that
    a packed step holds at most `cap` samples (or over `micro_batches` when given). Returns (a, samples per level in one
    micro-batch, total samples per level over the a micro-batches), or None for unusable inputs."""
    values = list(V) + list(C) + [target_var]
    if not V or len(V) != len(C) or not all(math.isfinite(x) and x > 0 for x in values):
        return None
    scale = sum(math.sqrt(v * c) for v, c in zip(V, C)) / target_var
    total = [max(1, math.ceil(scale * math.sqrt(v / c))) for v, c in zip(V, C)]
    a = micro_batches or max(1, math.ceil(sum(total) / cap))
    micro = [max(1, math.ceil(n / a)) for n in total]
    return a, micro, [a * n for n in micro]


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float:
    """Peak allocated GPU memory (GB) since the last reset_peak_memory; NaN without a GPU."""
    return torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else float("nan")


def time_packed_steps(
    model: nn.Module,
    level_weights: list[float],
    exit_weights: list[float],
    loss_kwargs: dict,
    autocast: Callable,
    base_depth: int,
    batch: tuple[torch.Tensor, ...],
    allocations: list[list[int]],
    params: list[nn.Parameter],
    device: torch.device,
    reps: int,
    memory: list[float] | None = None,
) -> list[float]:
    """Milliseconds of one real MLMC training step (all levels packed into one micro-batch of sum(N) samples, separate level
    forwards, one summed loss, one backward) for every allocation; NaN where the GPU runs out of memory. With a `memory`
    list, the peak GPU memory (GB) of each allocation's timing is appended to it (NaN on the CPU)."""
    out = []
    for sizes in allocations:
        reset_peak_memory(device)
        objective = MLMCObjective(model, list(sizes), level_weights, base_depth, exit_weights, **loss_kwargs)
        packed = tile_batch(batch, sum(sizes))

        def fn():
            with autocast():
                return objective(*packed)[0]

        try:
            out.append(time_per_sample(lambda _: fn, 1, params, device, reps))  # n = 1: ms per call
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            out.append(float("nan"))
        if memory is not None:
            memory.append(peak_memory_gb(device))
    return out


def budget_sweep(
    V: list[float],
    V_ref: float,
    C_ref_ms: float,
    block_costs: list[float],
    block_ref: float,
    time_level: Callable[[int, int], tuple[float, int]],
    budgets: list[int],
    max_forward: int = 128,
    iterations: int = 3,
) -> list[dict[str, float]]:
    """What a bigger compute budget buys: for each budget B (a baseline batch, in samples, whose compute is spent) the
    equal-cost allocation of the levels, each level run in forward batches of up to max_forward samples, with the costs
    timed at those batch sizes. The reference stays timed at the baseline's own micro-batch (C_ref_ms).

    The allocation depends on the costs and the costs on the batch sizes, so it is iterated: start from block costs, time
    the levels, re-spend the budget (measured, B * C_ref_ms) with the timed costs, repeat. time_level(j, n) returns (ms per
    sample, the batch size actually timed) for level j at forward batch n. Returns one row per budget, with the micro-batch
    each level runs at (forward_batch) and how many forward calls it takes per step (forward_calls). At a small budget the
    one-sample-per-level minimum can overspend (cost_vs_baseline above 1).
    """
    rows = []
    for budget in budgets:
        sizes = equal_cost_sizes(V, block_costs, budget * block_ref) or default_level_allocation(max(budget, len(V)), len(V))
        for it in range(iterations + 1):
            forward = [min(n, max_forward) for n in sizes]
            timed = [time_level(j, f) for j, f in enumerate(forward)]
            C, used = [t[0] for t in timed], [t[1] for t in timed]
            if it == iterations:
                break
            new = equal_cost_sizes(V, C, budget * C_ref_ms)
            if not new or new == sizes:
                break
            sizes = new
        n = [float(x) for x in sizes]
        root = giles_root(V, C, V_ref, C_ref_ms)
        rows.append({
            "budget": float(budget), "total_samples": float(sum(sizes)), "samples": [float(x) for x in sizes],
            "forward_batch": [float(u) for u in used],  # the micro-batch each level runs at
            "forward_calls": [float(math.ceil(x / u)) for x, u in zip(sizes, used)],  # forward calls per level per step
            "cost_ms_per_sample": C,
            "cost_vs_baseline": sum(a * c for a, c in zip(n, C)) / (budget * C_ref_ms),
            "root": root, "ideal_speedup": 1.0 / root**2 if root > EPS else float("nan"),
            "variance_gain": realized_gain(V, C, n, V_ref, C_ref_ms),
            "blocks_variance_gain": realized_gain(V, block_costs, n, V_ref, block_ref),
        })
    return rows


def measure(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    level_weights: list[float],
    exit_weights: list[float],
    loss_kwargs: dict,
    device: torch.device,
    autocast: Callable,
    timing_sizes: list[int] | Callable[[list[float]], list[int]] | None = None,
    ref_batch: int | None = None,
    timing_reps: int = 3,
    cost_cache: CostCache | None = None,
    base_depth: int = 1,
    common_noise: bool = True,
    deep_exit_weights: list[float] | None = None,
    max_chunk: list[int] | None = None,
) -> Measurements:
    """Measure V and C for every level term and for the baseline objective.

    deep_exit_weights: also measure V and C of the deep-supervised baseline objective with these exit weights (the second
    reference of a final-exit-only telescope, which the report compares against too).

    The telescope covers the depths base_depth..K (K = len(level_weights)). Its base level runs base_depth loops
    and reads all exits up to base_depth (see mlmc_base_term); the others are the usual corrections.

    V comes from the R replicate batches (images, text, text_mask). With common_noise (default) every term of a replicate
    (all levels and the reference) draws the same t, noise and label drop, so the heavy samples hit the numerator and
    the denominator of every ratio together and most of the estimation noise cancels (common random numbers); each
    term's variance stays unbiased. Without it each term draws its own. C is timed separately, after V, at the batch sizes that really run: level k at
    timing_sizes[k-1] samples (its share of the step batch) and the reference at ref_batch (the full
    per-GPU batch; both are halved on an out-of-memory error, and the sizes actually used are recorded).
    timing_sizes may also be a function of the measured variances (see timing_sizes_from_variances).
    With cost_cache the timing is skipped and the cached costs are reused (they change slowly); a callable timing_sizes
    still re-derives the allocation (alloc_sizes) from the current variances, which move faster than the costs.
    max_chunk: the most samples per forward of each level (level-wise micro-batching). The samples per level are then
    alloc_sizes, but each level is timed at its largest chunk, as it really runs, so the cost per sample is that of a
    chunk and a whole-step count too big for the GPU no longer forces a halving. Without it a level is timed at its
    whole count. Sizes default to the replicate batch size.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    samples = batches[0][0].shape[0]
    depths = list(range(base_depth, len(level_weights) + 1))
    num_levels = len(depths)
    out = Measurements(samples_per_replicate=samples)

    made = make_losses(model, level_weights, exit_weights, loss_kwargs, autocast, base_depth, deep_exit_weights)
    level_loss, ref_loss = made[:2]
    deep_loss = made[2] if deep_exit_weights is not None else None

    base_seed = int(torch.randint(0, 2**31 - 1, (1,)))
    seeds = [base_seed + r if common_noise else None for r in range(len(batches))]
    for depth in depths:
        v, g2, losses = replicate_gradient_variance(
            params, [with_seed(level_loss(depth, b), seeds[r], device) for r, b in enumerate(batches)], samples)
        out.V.append(v), out.grad_sq.append(g2)
        t = torch.tensor(losses)
        out.loss_mean.append(float(t.mean())), out.loss_var.append(float(t.var(unbiased=True)) * samples)
    out.V_ref, out.grad_sq_ref, _ = replicate_gradient_variance(
        params, [with_seed(ref_loss(b), seeds[r], device) for r, b in enumerate(batches)], samples)
    if deep_loss is not None:
        out.V_deep, out.grad_sq_deep, _ = replicate_gradient_variance(
            params, [with_seed(deep_loss(b), seeds[r], device) for r, b in enumerate(batches)], samples)

    if cost_cache is not None:
        out.C_ms, out.C_ref_ms = list(cost_cache.C_ms), cost_cache.C_ref_ms
        if deep_loss is not None:
            out.C_deep_ms = cost_cache.C_deep_ms
        out.timing_sizes, out.ref_batch = list(cost_cache.timing_sizes), cost_cache.ref_batch
        out.timing_wanted = list(cost_cache.timing_wanted or cost_cache.timing_sizes)
        out.alloc_sizes = [float(n) for n in timing_sizes(out.V)] if callable(timing_sizes) else list(
            cost_cache.alloc_sizes or cost_cache.timing_sizes)
        return out
    sizes = list(timing_sizes(out.V)) if callable(timing_sizes) else list(timing_sizes or [samples] * num_levels)
    wanted = [max(split_evenly(n, c)) for n, c in zip(sizes, max_chunk)] if max_chunk else list(sizes)
    used = []
    for j, depth in enumerate(depths):
        ms, n = time_per_sample_fit(lambda n, depth=depth: level_loss(depth, tile_batch(batches[0], n)),
                                    wanted[j], params, device, timing_reps)
        out.C_ms.append(ms), used.append(float(n))
    out.C_ref_ms, ref_used = time_per_sample_fit(lambda n: ref_loss(tile_batch(batches[0], n)),
                                                 ref_batch or samples, params, device, timing_reps)
    out.timing_sizes, out.ref_batch = used, float(ref_used)
    out.alloc_sizes, out.timing_wanted = [float(n) for n in sizes], [float(n) for n in wanted]
    if deep_loss is not None:
        out.C_deep_ms, _ = time_per_sample_fit(lambda n: deep_loss(tile_batch(batches[0], n)),
                                               ref_batch or samples, params, device, timing_reps)
    return out


def reduce_mean(meas: Measurements, num_levels: int, device: torch.device) -> Measurements:
    """Average the raw measurements over ranks (each rank saw different data)."""
    if not (dist.is_available() and dist.is_initialized()):
        return meas
    t = torch.tensor(meas.flat(), dtype=torch.float64, device=device)
    dist.all_reduce(t)
    t /= dist.get_world_size()
    return Measurements.unflat(t.tolist(), num_levels, meas.samples_per_replicate)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    meas: Measurements,
    block_costs: list[float],
    block_ref: float,
    level_sizes: list[int] | None = None,
    world_accum: int = 1,
    timing_sizes: list[int] | None = None,
    ref_batch: int | None = None,
    depths: list[int] | None = None,
    budget_batch: int | None = None,
    deep_block_ref: float | None = None,
) -> dict[str, float]:
    """Flatten the measurements into the logged metrics (namespace mlmc_diag/ and mlmc_loss_proxy/).

    budget_batch: with --baseline, the baseline batch whose compute the hypothetical allocation (the timing sizes) is
    meant to match; enables the equal_cost/ metrics.

    depths: the loop depth of each level (default 1..K); metrics are named level{depth}, so a telescope whose
    base level runs 2 loops has level2 (base), level3 and level4 and no level1.

    deep_block_ref: blocks per sample of the deep-supervised baseline. With it and a measured meas.V_deep (a telescope of the
    final exit only), the metrics are repeated against that reference as mlmc_diag/vs_deep/... (and vs_deep/blocks/...): the
    variance and cost ratios of the final-only terms to the full deep-supervised objective L_K + mean of the shallower exits.
    The two estimate different gradients (the final-only objective is not the deep-supervised one), so these ratios say what
    the final-only telescope costs against the baseline that is actually trained, not that it is unbiased for it.

    level_sizes: current samples per level and micro-batch (mlmc mode); world_accum = GPUs x
    accumulation steps, turning them into global samples per optimizer step. timing_sizes / ref_batch:
    the batch sizes the costs were timed at (logged for transparency).
    """
    k = len(meas.V)
    depths = depths or list(range(1, k + 1))
    m: dict[str, float] = {}
    for tag, V, C, V_ref, C_ref in (
        ("", meas.V, meas.C_ms, meas.V_ref, meas.C_ref_ms),
        ("blocks/", meas.V, block_costs, meas.V_ref, block_ref),
    ):
        root = giles_root(V, C, V_ref, C_ref)
        m[f"mlmc_diag/{tag}giles_root"] = root
        m[f"mlmc_diag/{tag}ideal_speedup"] = 1.0 / root**2 if root > EPS else float("nan")
        m[f"mlmc_diag/{tag}speedup_possible"] = float(root < 1.0) if math.isfinite(root) else 0.0
        alloc = giles_allocation(V, C)
        for lvl in range(k):
            pre = f"mlmc_diag/{tag}level{depths[lvl]}"
            m[f"{pre}/variance_ratio"] = V[lvl] / V_ref if V_ref > 0 else float("nan")
            m[f"{pre}/cost_ratio"] = C[lvl] / C_ref if C_ref > 0 else float("nan")
            m[f"{pre}/giles_contribution"] = (
                math.sqrt(m[f"{pre}/variance_ratio"] * m[f"{pre}/cost_ratio"])
                if V_ref > 0 and C_ref > 0 and V[lvl] >= 0 else float("nan")
            )
            m[f"{pre}/optimal_sample_fraction"] = alloc["sample_fraction"][lvl]
            m[f"{pre}/optimal_work_share"] = alloc["work_share"][lvl]
    for lvl in range(k):
        pre = f"mlmc_diag/level{depths[lvl]}"
        m[f"{pre}/grad_variance"] = meas.V[lvl]
        m[f"{pre}/cost_ms_per_sample"] = meas.C_ms[lvl]
        m[f"{pre}/cost_blocks"] = block_costs[lvl]
        m[f"{pre}/grad_norm_sq"] = meas.grad_sq[lvl]
        if lvl >= 1:  # decay of the correction variance and growth of the cost with the level
            m[f"{pre}/variance_ratio_to_previous"] = meas.V[lvl] / meas.V[lvl - 1] if meas.V[lvl - 1] > 0 else float("nan")
            m[f"{pre}/cost_ratio_to_previous"] = meas.C_ms[lvl] / meas.C_ms[lvl - 1] if meas.C_ms[lvl - 1] > 0 else float("nan")
        m[f"mlmc_loss_proxy/level{depths[lvl]}/mean"] = meas.loss_mean[lvl]
        m[f"mlmc_loss_proxy/level{depths[lvl]}/variance"] = meas.loss_var[lvl]
    levels = depths[1:]
    m["mlmc_diag/fit/beta"] = -log2_slope(levels, meas.V[1:])  # V_l ~ 2^(-beta l) over the corrections
    m["mlmc_diag/fit/gamma"] = log2_slope(levels, meas.C_ms[1:])  # C_l ~ 2^(gamma l)
    m["mlmc_diag/fit/gamma_blocks"] = log2_slope(levels, block_costs[1:])
    m["mlmc_diag/reference/grad_variance"] = meas.V_ref
    m["mlmc_diag/reference/cost_ms_per_sample"] = meas.C_ref_ms
    m["mlmc_diag/reference/cost_blocks"] = block_ref
    m["mlmc_diag/reference/grad_norm_sq"] = meas.grad_sq_ref
    m["mlmc_diag/diag_batch_per_replicate"] = float(meas.samples_per_replicate)
    for lvl, n in enumerate(timing_sizes or meas.timing_sizes):
        m[f"mlmc_diag/level{depths[lvl]}/timing_batch"] = float(n)  # batch size at which the cost was timed
    wanted = meas.timing_wanted or meas.timing_sizes
    for lvl, (asked, used) in enumerate(zip(wanted, meas.timing_sizes)):
        m[f"mlmc_diag/level{depths[lvl]}/timing_halved"] = float(used < asked)  # 1: out of memory, the cost is of a smaller batch
    m["mlmc_diag/timing_halved_levels"] = float(sum(used < asked for asked, used in zip(wanted, meas.timing_sizes)))
    if ref_batch or meas.ref_batch:
        m["mlmc_diag/reference/timing_batch"] = float(ref_batch or meas.ref_batch)
    if level_sizes is None and budget_batch and meas.timing_sizes and meas.C_ref_ms > 0:
        # The timing sizes are the hypothetical allocation: what it costs against the baseline's batch, and the variance
        # reduction it gives at its own cost (above 1: MLMC wins). Costs were timed at exactly these sizes.
        n = list(meas.alloc_sizes or meas.timing_sizes)  # the allocation; each level's cost was timed at (a chunk of) its count
        for tag, C, C_ref in (("", meas.C_ms, meas.C_ref_ms), ("blocks/", block_costs, block_ref)):
            m[f"mlmc_diag/{tag}equal_cost/cost_vs_baseline"] = sum(a * c for a, c in zip(n, C)) / (budget_batch * C_ref)
            m[f"mlmc_diag/{tag}equal_cost/variance_gain"] = realized_gain(meas.V, C, n, meas.V_ref, C_ref)
        m["mlmc_diag/equal_cost/total_samples"] = float(sum(n))
        m["mlmc_diag/equal_cost/baseline_batch"] = float(budget_batch)
        for lvl in range(k):
            m[f"mlmc_diag/level{depths[lvl]}/equal_cost_samples"] = n[lvl]
    if deep_block_ref and math.isfinite(meas.V_deep):
        deep = "mlmc_diag/vs_deep"
        for tag, C, C_ref in (("", meas.C_ms, meas.C_deep_ms), ("blocks/", block_costs, deep_block_ref)):
            root = giles_root(meas.V, C, meas.V_deep, C_ref)
            m[f"{deep}/{tag}giles_root"] = root
            m[f"{deep}/{tag}ideal_speedup"] = 1.0 / root**2 if root > EPS else float("nan")
            m[f"{deep}/{tag}speedup_possible"] = float(root < 1.0) if math.isfinite(root) else 0.0
            for lvl in range(k):
                pre = f"{deep}/{tag}level{depths[lvl]}"
                m[f"{pre}/variance_ratio"] = meas.V[lvl] / meas.V_deep if meas.V_deep > 0 else float("nan")
                m[f"{pre}/cost_ratio"] = C[lvl] / C_ref if C_ref > 0 else float("nan")
                m[f"{pre}/giles_contribution"] = (
                    math.sqrt(m[f"{pre}/variance_ratio"] * m[f"{pre}/cost_ratio"])
                    if meas.V_deep > 0 and C_ref > 0 and meas.V[lvl] >= 0 else float("nan")
                )
        m[f"{deep}/reference/grad_variance"] = meas.V_deep
        m[f"{deep}/reference/cost_ms_per_sample"] = meas.C_deep_ms
        m[f"{deep}/reference/cost_blocks"] = deep_block_ref
        m[f"{deep}/reference/grad_norm_sq"] = meas.grad_sq_deep
        m[f"{deep}/reference/variance_vs_final_only"] = meas.V_deep / meas.V_ref if meas.V_ref > 0 else float("nan")
        if level_sizes is None and budget_batch and meas.timing_sizes and meas.C_deep_ms > 0:
            n = list(meas.alloc_sizes or meas.timing_sizes)
            for tag, C, C_ref in (("", meas.C_ms, meas.C_deep_ms), ("blocks/", block_costs, deep_block_ref)):
                m[f"{deep}/{tag}equal_cost/cost_vs_baseline"] = sum(a * c for a, c in zip(n, C)) / (budget_batch * C_ref)
                m[f"{deep}/{tag}equal_cost/variance_gain"] = realized_gain(meas.V, C, n, meas.V_deep, C_ref)
        if level_sizes is not None:
            n = [float(s) * world_accum for s in level_sizes]
            m[f"{deep}/realized/variance_gain_at_equal_cost"] = realized_gain(meas.V, meas.C_ms, n, meas.V_deep, meas.C_deep_ms)
            m[f"{deep}/realized/variance_gain_at_equal_cost_blocks"] = realized_gain(meas.V, block_costs, n, meas.V_deep, deep_block_ref)
            m[f"{deep}/realized/cost_ratio_per_sample"] = (
                sum(a * c for a, c in zip(n, meas.C_ms)) / sum(n) / meas.C_deep_ms if meas.C_deep_ms > 0 else float("nan")
            )
            m[f"{deep}/realized/cost_ratio_per_sample_blocks"] = sum(a * c for a, c in zip(n, block_costs)) / sum(n) / deep_block_ref
    if level_sizes is not None:
        n = [float(s) * world_accum for s in level_sizes]
        total = sum(n)
        for lvl in range(k):
            m[f"mlmc_diag/level{depths[lvl]}/current_sample_fraction"] = n[lvl] / total
        m["mlmc_diag/realized/variance_gain_at_equal_cost"] = realized_gain(meas.V, meas.C_ms, n, meas.V_ref, meas.C_ref_ms)
        m["mlmc_diag/realized/variance_gain_at_equal_cost_blocks"] = realized_gain(meas.V, block_costs, n, meas.V_ref, block_ref)
        m["mlmc_diag/realized/cost_ratio_per_sample"] = (
            sum(a * c for a, c in zip(n, meas.C_ms)) / total / meas.C_ref_ms if meas.C_ref_ms > 0 else float("nan")
        )
        m["mlmc_diag/realized/cost_ratio_per_sample_blocks"] = sum(a * c for a, c in zip(n, block_costs)) / total / block_ref
    return m


EMA_SUFFIXES = ("variance_ratio", "cost_ratio", "giles_contribution", "giles_root", "variance_gain",
                "variance_gain_at_equal_cost", "optimal_sample_fraction")


class LogEma:
    """Exponential moving average of the diagnostic ratios, taken in log space (they are multiplicative and noisy), as
    mlmc_diag_ema/... metrics. Non-positive or non-finite values are skipped; the ideal speedups follow from the averaged roots."""

    def __init__(self, decay: float):
        self.decay, self.state = decay, {}

    def update(self, report: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in report.items():
            if not key.endswith(EMA_SUFFIXES) or not (math.isfinite(value) and value > 0):
                continue
            previous = self.state.get(key)
            self.state[key] = math.log(value) if previous is None else self.decay * previous + (1 - self.decay) * math.log(value)
            out[key.replace("mlmc_diag/", "mlmc_diag_ema/", 1)] = math.exp(self.state[key])
        for tag in ("", "blocks/", "vs_deep/", "vs_deep/blocks/"):
            root = out.get(f"mlmc_diag_ema/{tag}giles_root")
            if root:
                out[f"mlmc_diag_ema/{tag}ideal_speedup"] = 1.0 / root**2
        return out

# ---------------------------------------------------------------------------
# Empirical test of the estimators
# ---------------------------------------------------------------------------


class SampleStream:
    """take(n) gives n fresh (images, text, text_mask) samples, pulling micro-batches from next_batch as needed."""

    def __init__(self, next_batch: Callable[[], tuple[torch.Tensor, ...]]):
        self.next_batch, self.buffer = next_batch, None

    def take(self, n: int) -> tuple[torch.Tensor, ...]:
        parts, have = [], 0
        while have < n:
            if self.buffer is None or self.buffer[0].shape[0] == 0:
                self.buffer = self.next_batch()
            k = min(n - have, self.buffer[0].shape[0])
            parts.append(tuple(x[:k] for x in self.buffer))
            self.buffer = tuple(x[k:] for x in self.buffer)
            have += k
        return tuple(torch.cat([part[i] for part in parts]) for i in range(len(parts[0])))


def chunk_sizes(n: int, cap: int) -> list[int]:
    """n samples in forward chunks of at most cap."""
    cap = max(1, min(cap, n))
    return [cap] * (n // cap) + ([n % cap] if n % cap else [])


def _accumulate(total: list[torch.Tensor], loss: torch.Tensor, params: list[nn.Parameter], scale: float) -> None:
    for acc, g in zip(total, torch.autograd.grad(loss * scale, params, allow_unused=True)):
        if g is not None:
            acc += g.float()


def baseline_gradient_estimate(ref_loss: Callable, stream: SampleStream, params: list[nn.Parameter], batch: int, chunk: int):
    """One gradient estimate of the baseline objective from `batch` fresh samples, run in forward chunks of `chunk`."""
    total = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    for n in chunk_sizes(batch, chunk):
        _accumulate(total, ref_loss(stream.take(n))(), params, n / batch)
    return total


def telescope_gradient_estimate(
    level_loss: Callable, depths: list[int], stream: SampleStream, params: list[nn.Parameter],
    samples: list[int], forward: list[int],
):
    """One gradient estimate of the telescope: level k gets samples[k] fresh samples (independent of the other levels),
    run in forward chunks of forward[k]; each chunk's weighted mean loss counts n / samples[k] of the level's term."""
    total = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    for depth, count, cap in zip(depths, samples, forward):
        for n in chunk_sizes(int(count), int(cap)):
            _accumulate(total, level_loss(depth, stream.take(n))(), params, n / int(count))
    return total


def estimator_statistics(
    estimate: Callable[[], list[torch.Tensor]], num_estimates: int, device: torch.device
) -> dict:
    """Run an estimator num_estimates times. Returns the empirical variance of the estimates (trace of their covariance,
    from running sums, so only a few gradient-sized buffers are kept), the same from the even and the odd estimates (a
    stability check), the mean time per estimate (synchronized) and the mean estimate."""
    sums = [None, None]
    sq, counts, times = [0.0, 0.0], [0, 0], []
    for i in range(num_estimates):
        _sync(device)
        start = time.perf_counter()
        grads = estimate()
        _sync(device)
        times.append(time.perf_counter() - start)
        half = i % 2
        sums[half] = [g.clone() for g in grads] if sums[half] is None else [a.add_(g) for a, g in zip(sums[half], grads)]
        sq[half] += sum(float(g.pow(2).sum()) for g in grads)
        counts[half] += 1
        del grads

    def trace_cov(sum_vec, sq_sum, n):
        if sum_vec is None or n < 2:
            return float("nan")
        return max((sq_sum - sum(float(t.pow(2).sum()) for t in sum_vec) / n) / (n - 1), 0.0)

    both = [a + b for a, b in zip(sums[0], sums[1])] if sums[1] is not None else sums[0]
    n_all = counts[0] + counts[1]
    return {
        "variance": trace_cov(both, sq[0] + sq[1], n_all),
        "variance_even": trace_cov(sums[0], sq[0], counts[0]), "variance_odd": trace_cov(sums[1], sq[1], counts[1]),
        "time_s": sum(times) / len(times), "num_estimates": n_all, "mean": [t / n_all for t in both],
    }


def compare_estimators(baseline: dict, telescope: dict, predicted_baseline: float, predicted_telescope: float) -> dict[str, float]:
    """Empirical comparison of the baseline and telescope estimators from estimator_statistics: variance and time ratios,
    the efficiency gain (variance times time, higher is better for the telescope), the same predicted from the per-level
    variances, and a bias check (the distance between the two mean estimates against what noise alone explains: about 1
    when both are unbiased for the same gradient, much larger when not)."""
    diff = sum(float((a - b).pow(2).sum()) for a, b in zip(baseline["mean"], telescope["mean"]))
    expected = baseline["variance"] / baseline["num_estimates"] + telescope["variance"] / telescope["num_estimates"]
    return {
        "variance_baseline": baseline["variance"], "variance_telescope": telescope["variance"],
        "time_baseline_s": baseline["time_s"], "time_telescope_s": telescope["time_s"],
        "variance_ratio": baseline["variance"] / telescope["variance"],  # above 1: the telescope's estimate is less noisy
        "time_ratio": telescope["time_s"] / baseline["time_s"],  # near 1 when the budget was matched
        "efficiency_gain": (baseline["variance"] * baseline["time_s"]) / (telescope["variance"] * telescope["time_s"]),
        "predicted_variance_ratio": predicted_baseline / predicted_telescope,
        "bias_statistic": diff / expected if expected > 0 else float("nan"),
        "variance_ratio_even": baseline["variance_even"] / telescope["variance_even"],
        "variance_ratio_odd": baseline["variance_odd"] / telescope["variance_odd"],
    }
