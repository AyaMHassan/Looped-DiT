"""Train Looped-DiT.

Pretraining:
    torchrun --nproc_per_node=8 -m looped_dit.train --config configs/b32_pretrain.yml \
        --output-dir outputs/b32_pretrain

Fine-tuning continues the step counter, weights, EMA and optimizer of a pretrained checkpoint:
    torchrun --nproc_per_node=8 -m looped_dit.train --config configs/b32_finetune.yml \
        --output-dir outputs/b32_finetune --init-from outputs/b32_pretrain/checkpoints/checkpoint_0250000.pt

A run resumes automatically from the newest checkpoint in its output directory.

Training objective: --baseline (default) runs every sample at full depth with deep supervision;
--mlmc telescopes the same objective over loop depth (level k = depth k) with coupled exit pairs:
    torchrun --nproc_per_node=8 -m looped_dit.train --config configs/b16_pretrain.yml \
        --output-dir outputs/b16_mlmc --mlmc --mlmc-levels 4 --mlmc-batch 4 2 1 1
(--mlmc-base-depth 3 trains only the depths 3 and 4: --mlmc-batch then takes two entries.)
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from .config import TrainConfig
from .data import make_loader, make_packed_loader
from . import wandb_log
from .diffusion import (
    MLMCChunkObjective,
    MLMCObjective,
    corrupt,
    deep_supervision_weights,
    default_level_allocation,
    mlmc_chunk_plan,
    final_exit_weights,
    euler_sample,
    mlmc_level_weights,
    training_loss,
)
from .fid import check_fid_setup, run_fid
from .mlmc_diagnostics import (
    baseline_block_cost,
    build_report,
    level_block_costs,
    LogEma,
    measure,
    reduce_mean,
    replicate_batches,
    reference_block_cost,
    timing_sizes_from_variances,
)
from .model import LoopedMMDiT
from .pipeline import TextEncoder
from .utils import (
    amp_dtype,
    atomic_save,
    init_distributed,
    is_main,
    learning_rate,
    rank,
    seed_everything,
    update_ema,
    world_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Looped-DiT.")
    parser.add_argument("--config", required=True, help="training config (YAML)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-from", help="checkpoint to start from, e.g. the pretrained model for fine-tuning")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", action="store_true", help="train every sample at full depth (default; sets train_mode)")
    mode.add_argument("--mlmc", action="store_true", help="multilevel Monte Carlo over loop depth (sets train_mode)")
    parser.add_argument("--mlmc-levels", type=int, help="top depth K of the MLMC telescope (default: num_loops)")
    parser.add_argument("--mlmc-base-depth", type=int,
                        help="depth of the shallowest MLMC level (default 1): levels base..K, one --mlmc-batch entry each")
    parser.add_argument("--mlmc-batch", type=int, nargs="+", metavar="N",
                        help="samples per level, per GPU and micro-batch, shallowest level first (default: micro_batch_size halved per level)")
    parser.add_argument("--loop-split", type=int, nargs=3, metavar=("PRE", "CORE", "POST"),
                        help="blocks before / inside / after the loop (sets loop_split; default 6 5 6)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--wandb-env", metavar="FILE", help="dotenv-style file with WANDB_API_KEY etc. (sets wandb_env)")
    parser.add_argument("--wandb-project", help="sets wandb_project")
    parser.add_argument("--wandb-entity", help="sets wandb_entity")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], help="sets wandb_mode")
    parser.add_argument("--fid-no-ema", action="store_true", help="run the periodic FID on the raw model instead of the EMA (sets fid_use_ema=false)")
    return parser.parse_args()


def checkpoints(directory: Path) -> list[Path]:
    return sorted(directory.glob("checkpoint_*.pt"))


def save_checkpoint(directory: Path, step: int, model, ema, optimizer, cfg: TrainConfig) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
    }
    atomic_save(state, directory / f"checkpoint_{step:07d}.pt")
    old = checkpoints(directory)[: -cfg.keep_last]
    for path in old:
        if cfg.ckpt_keep_every <= 0 or int(path.stem.split("_")[1]) % cfg.ckpt_keep_every:
            path.unlink()


def load_checkpoint(path: Path, cfg: TrainConfig, model, ema, optimizer) -> int:
    """Restore weights, EMA, optimizer state (when present) and the step counter."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    saved, current = ckpt["config"], cfg.model_kwargs()
    mismatch = {k: (saved.get(k), v) for k, v in current.items() if saved.get(k) != (list(v) if isinstance(v, tuple) else v)}
    if mismatch:
        raise ValueError(f"{path} was trained with a different architecture (checkpoint, config): {mismatch}")
    # A checkpoint with only EMA weights initializes both copies from them.
    model.load_state_dict(ckpt.get("model", ckpt["ema"]))
    ema.load_state_dict(ckpt["ema"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        for group in optimizer.param_groups:  # hyperparameters come from the config, not the checkpoint
            group.update(betas=(0.9, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    if is_main():
        state = "" if "optimizer" in ckpt else "; no optimizer state, Adam starts fresh"
        print(f"loaded {path} (step {ckpt['step']}{state})", flush=True)
    return int(ckpt["step"])


@torch.no_grad()
@torch.no_grad()
def exit_visualization(model, images, text, text_mask, cfg, device, autocast, num_levels, n: int = 4):
    """Rows of [x0 | x_t | prediction after 1 loop | ... | prediction after K loops] at fixed noise.
    Returns (the picture as a [-1, 1] tensor, scalar statistics of every panel)."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(0)
        x_t, _, _, mask = corrupt(images[:n], text_mask[:n], cfg.noise_scale, cfg.t_logit_mean, cfg.t_logit_std, 0.0)
    with autocast():
        final, exits = model(x_t, text[:n], mask, num_loops=num_levels, exit_loops=tuple(range(1, num_levels)))
    preds = [exits[r].float() for r in range(1, num_levels)] + [final.float()]
    stats = wandb_log.tensor_stats("viz/x0", images[:n])
    for r, pred in enumerate(preds, start=1):
        stats.update(wandb_log.tensor_stats(f"viz/pred_loop{r}", pred))
    panels = [images[:n], x_t.clamp(-1, 1)] + preds
    return torch.cat([p.float().cpu() for p in panels], dim=3), stats


def main() -> None:
    args = parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)}
    if args.baseline or args.mlmc:
        overrides["train_mode"] = "mlmc" if args.mlmc else "baseline"
    if args.mlmc_levels is not None:
        overrides["mlmc_levels"] = args.mlmc_levels
    if args.loop_split is not None:
        overrides["loop_split"] = args.loop_split
    if args.mlmc_base_depth is not None:
        overrides["mlmc_base_depth"] = args.mlmc_base_depth
    if args.mlmc_batch is not None:
        overrides["mlmc_batch"] = args.mlmc_batch
    if args.fid_no_ema:
        overrides["fid_use_ema"] = False
    for flag in ("wandb_env", "wandb_project", "wandb_entity", "wandb_mode"):
        if getattr(args, flag) is not None:
            overrides[flag] = getattr(args, flag)
    cfg = TrainConfig.from_yaml(args.config, overrides)
    mlmc = cfg.train_mode == "mlmc"
    if mlmc:
        level_sizes = list(cfg.mlmc_batch) or default_level_allocation(cfg.micro_batch_size, cfg.mlmc_level_count())
        cfg.mlmc_levels, cfg.mlmc_batch = cfg.num_mlmc_levels(), level_sizes  # recorded in the saved config
    micro_batch = sum(cfg.mlmc_batch) if mlmc else cfg.micro_batch_size  # samples read per micro-batch and GPU
    device = init_distributed()
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    per_step = micro_batch * world_size()
    if cfg.batch_size % per_step:
        raise ValueError(f"batch_size {cfg.batch_size} is not a multiple of micro-batch {micro_batch} x GPUs = {per_step}")
    accum = cfg.batch_size // per_step
    level_wise = mlmc and cfg.mlmc_level_batching
    if level_wise and accum != 1:
        raise ValueError(f"mlmc_level_batching runs a whole step per GPU at once: batch_size must be sum(mlmc_batch) x GPUs "
                         f"= {per_step}, got {cfg.batch_size}")
    dtype = amp_dtype(cfg.amp_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(cfg.seed)  # same initialization on every rank
    model = LoopedMMDiT(**cfg.model_kwargs()).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, betas=(0.9, cfg.adam_beta2), weight_decay=cfg.weight_decay, fused=True
    )
    step = 0
    existing = checkpoints(ckpt_dir)
    resume = existing[-1] if existing else Path(args.init_from) if args.init_from else None
    if resume is not None:
        step = load_checkpoint(resume, cfg, model, ema, optimizer)
    seed_everything(cfg.seed + rank() + step)  # a resumed run does not replay the same noise
    loss_kwargs = dict(
        noise_scale=cfg.noise_scale,
        t_logit_mean=cfg.t_logit_mean,
        t_logit_std=cfg.t_logit_std,
        label_drop_rate=cfg.label_drop_rate,
    )
    exit_weights = deep_supervision_weights(cfg.num_loops, cfg.deep_supervision_weighting) if cfg.deep_supervision else None
    if mlmc:
        # One module = one forward per micro-batch, which DDP requires; it wraps the same parameters.
        # deep_supervision false: the final-only telescope F_1 + sum_k (F_k - S_{k-1}) = F_K, every weight 1
        if cfg.deep_supervision:
            level_weights = mlmc_level_weights(cfg.mlmc_levels, cfg.deep_supervision_weighting)
            mlmc_exit_weights = deep_supervision_weights(cfg.mlmc_levels, cfg.deep_supervision_weighting)
        else:
            mlmc_exit_weights, level_weights = final_exit_weights(cfg.mlmc_levels)
        if level_wise:  # one module per chunk: DDP sees one forward per backward, the step syncs on the last chunk
            objective = MLMCChunkObjective(model, level_weights, cfg.mlmc_base_depth, mlmc_exit_weights, **loss_kwargs)
        else:
            objective = MLMCObjective(model, cfg.mlmc_batch, level_weights, cfg.mlmc_base_depth, mlmc_exit_weights, **loss_kwargs)
        net = DDP(objective, device_ids=[device.index], gradient_as_bucket_view=True, bucket_cap_mb=100) if world_size() > 1 else objective
        objective_desc = (f"mlmc depths {cfg.mlmc_base_depth}..{cfg.mlmc_levels}{'' if cfg.deep_supervision else ' (final exit only)'}, "
                          f"samples per level {cfg.mlmc_batch}, correction weights {level_weights}")
    else:
        net = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True, bucket_cap_mb=100) if world_size() > 1 else model
        objective_desc = f"baseline, exit weights {exit_weights}"

    text_encoder = TextEncoder(cfg.text_encoder, cfg.prompt_length, device)
    batches = iter(make_packed_loader(cfg, cfg.mlmc_batch, seed_offset=step) if mlmc else make_loader(cfg, seed_offset=step))
    if is_main():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{n_params / 1e6:.1f}M parameters, {world_size()} GPUs x micro-batch {micro_batch} "
              f"x {accum} accumulation steps = batch {cfg.batch_size}; {objective_desc}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    wandb_on = False
    if args.wandb and is_main():
        wandb_on = wandb_log.init_wandb(cfg, out_dir, out_dir.name, cfg.to_dict())
    if cfg.fid_every:
        check_fid_setup(cfg)

    # Cost model (transformer blocks per sample) for the compute axis and the Giles diagnostics.
    num_levels = cfg.num_mlmc_levels()
    pre, core, post = model.pre, model.core, model.post
    # The telescope the diagnostics evaluate: depths base_depth..K. --mlmc: the one it trains (mlmc_base_depth). --baseline:
    # diag_base_depth (default 1: levels 1..K; 2 gives levels 2..K, whose base level runs 2 loops and reads exit 1 too).
    base_depth = cfg.mlmc_base_depth if mlmc else max(1, min(cfg.diag_base_depth, num_levels - 1))
    diag_depths = list(range(base_depth, num_levels + 1))
    diag_final_only = cfg.diag_objective == "final" or (cfg.diag_objective == "auto" and not cfg.deep_supervision)
    if diag_final_only:  # no deep supervision: only the final exit has weight, so every telescope weight is 1
        diag_exit_weights, diag_level_weights = final_exit_weights(num_levels)
    else:
        diag_level_weights = mlmc_level_weights(num_levels, cfg.deep_supervision_weighting)
        diag_exit_weights = deep_supervision_weights(num_levels, cfg.deep_supervision_weighting)
    block_costs = level_block_costs(pre, core, post, num_levels, base_depth, diag_exit_weights)
    block_ref = reference_block_cost(pre, core, post, num_levels, diag_exit_weights)
    # A final-exit-only telescope is also measured against the deep-supervised baseline objective (mlmc_diag/vs_deep/...).
    deep_ref_weights = deep_supervision_weights(num_levels, cfg.deep_supervision_weighting) if diag_final_only else None
    deep_block_ref = reference_block_cost(pre, core, post, num_levels, deep_ref_weights) if diag_final_only else None
    if mlmc:
        blocks_per_sample = sum(n * c for n, c in zip(cfg.mlmc_batch, block_costs)) / sum(cfg.mlmc_batch)
    else:
        blocks_per_sample = baseline_block_cost(pre, core, post, cfg.num_loops, cfg.deep_supervision)
    diag_on = bool(cfg.diag_every) and num_levels >= 2
    # Costs are timed at the batch sizes a training step really runs. --mlmc: each level at its packed micro-batch share
    # (mlmc_batch). --baseline has no allocation: each level at the equal-cost optimal N_k, i.e. the samples that
    # the compute of the baseline's batch (diag_alloc_batch, default micro_batch_size) buys when spent optimally over the
    # levels (more samples in total than that batch, as the levels are cheaper), decided after the variances are
    # measured. The reference runs at the baseline's forward batch (diag_ref_batch, default micro_batch_size).
    alloc_batch = cfg.diag_alloc_batch or cfg.micro_batch_size
    chunk_plan: list[tuple[int, int, int, float]] = []
    if level_wise:
        # Each level in the fewest equal chunks that fit the activation budget (blocks x samples per forward).
        max_blocks = cfg.mlmc_max_blocks or cfg.micro_batch_size * block_ref
        max_chunk = [max(1, int(max_blocks // cost)) for cost in block_costs]
        chunk_plan = mlmc_chunk_plan(cfg.mlmc_batch, max_chunk)
        if is_main():
            sizes = [[b - a for i, a, b, _ in chunk_plan if i == lvl] for lvl in range(len(cfg.mlmc_batch))]
            print(f"level-wise micro-batching, forward sizes per level {sizes}: {len(chunk_plan)} forwards per step "
                  f"(baseline micro-batch budget {max_blocks:.0f} blocks x samples)", flush=True)
    # Level-wise MLMC runs each level in chunks, and the baseline diagnostics time the levels the same way (diag_level_chunks):
    # a level is timed at its largest chunk, not at its whole-step count. The packed --mlmc code runs levels whole.
    max_chunk = None
    if level_wise or (not mlmc and cfg.diag_level_chunks):
        chunk_blocks = cfg.mlmc_max_blocks or cfg.micro_batch_size * block_ref
        max_chunk = [max(1, int(chunk_blocks // cost)) for cost in block_costs]
    timing_sizes = list(cfg.mlmc_batch) if mlmc else (
        lambda V: timing_sizes_from_variances(V, block_costs, alloc_batch, len(diag_depths), block_ref)
    )
    ref_batch = cfg.diag_ref_batch or cfg.micro_batch_size
    diag_ema = LogEma(cfg.diag_ema_decay) if cfg.diag_ema_decay else None
    diag_cost_cache, diag_cost_step = None, 0  # costs of the last cost measurement and its step, reused in between
    autocast = lambda: torch.autocast("cuda", dtype=dtype)  # noqa: E731

    def to_device(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = batch["pixel_values"].to(device, non_blocking=True).float().mul_(1 / 127.5).add_(-1.0)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        text_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            text = text_encoder.encode(input_ids, text_mask).to(dtype)
        return images, text, text_mask

    net.train()
    clock = wandb_log.ActiveClock()
    metric_sums: dict[str, torch.Tensor] = {}  # summed over the micro-batches of a log interval (this rank)
    step_losses: list[torch.Tensor] = []  # per optimizer step, for the loss / grad-norm variance over a log interval
    step_grad_norms: list[torch.Tensor] = []
    micro_batches, last_time, last_step, data_seconds = 0, time.time(), step, 0.0
    start_step = step
    while step < cfg.num_steps:
        step_loss = 0.0
        if level_wise:  # the whole step of this GPU is one packed batch, run level by level in chunks
            fetched = time.perf_counter()
            batch = next(batches)
            data_seconds += time.perf_counter() - fetched
            images, text, text_mask = to_device(batch)
            for j, (lvl, a, b, fraction) in enumerate(chunk_plan):
                # The deepest level is last and touches every parameter, so DDP's reduction completes on its backward.
                no_sync = world_size() > 1 and j < len(chunk_plan) - 1
                with net.no_sync() if no_sync else contextlib.nullcontext():
                    with torch.autocast("cuda", dtype=dtype):
                        loss, metrics = net(images[a:b], text[a:b], text_mask[a:b], cfg.mlmc_base_depth + lvl, fraction)
                    loss.backward()
                step_loss = step_loss + metrics["loss"]
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0) + value
            micro_batches += 1
        for micro in range(0 if level_wise else accum):
            fetched = time.perf_counter()
            batch = next(batches)
            data_seconds += time.perf_counter() - fetched
            images, text, text_mask = to_device(batch)
            no_sync = world_size() > 1 and micro < accum - 1
            with net.no_sync() if no_sync else contextlib.nullcontext():
                with torch.autocast("cuda", dtype=dtype):
                    if mlmc:
                        loss, metrics = net(images, text, text_mask)
                    else:
                        loss, metrics = training_loss(net, images, text, text_mask, exit_weights, **loss_kwargs)
                (loss / accum).backward()
            step_loss = step_loss + metrics["loss"] / accum
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0) + value
            micro_batches += 1

        lr = learning_rate(step, cfg.learning_rate, cfg.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], cfg.max_grad_norm if cfg.max_grad_norm > 0 else math.inf
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_ema(ema, model, cfg.ema_decay)
        step += 1
        step_losses.append(step_loss)
        step_grad_norms.append(grad_norm.detach())

        if step % cfg.log_every == 0 and is_main():
            now = time.time()
            steps_per_sec = (step - last_step) / (now - last_time)
            means = {k: v.item() / micro_batches for k, v in metric_sums.items()}
            losses, norms = torch.stack(step_losses), torch.stack(step_grad_norms)
            log = {"step": step, **means, "lr": lr, "grad_norm": grad_norm.item(), "steps_per_sec": steps_per_sec}
            print(json.dumps(log), flush=True)
            if wandb_on:
                payload = {(k if k.startswith("mlmc/") else f"train/{k}"): v for k, v in means.items()}
                payload.update({
                    "train/lr": lr,
                    "train/grad_norm": grad_norm.item(),  # before clipping
                    "train/grad_norm_clipped": min(grad_norm.item(), cfg.max_grad_norm) if cfg.max_grad_norm > 0 else grad_norm.item(),
                    "train/loss_var": losses.var().item() if len(losses) > 1 else 0.0,  # over the steps of this interval
                    "train/grad_norm_var": norms.var().item() if len(norms) > 1 else 0.0,
                    "runtime/steps_per_second": steps_per_sec,
                    "runtime/samples_per_second": steps_per_sec * cfg.batch_size,
                    "runtime/max_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                    "timing/data_ms_per_step": 1000 * data_seconds / (step - last_step),
                    "compute/blocks_per_sample": blocks_per_sample,
                    "compute/cum_block_samples": blocks_per_sample * cfg.batch_size * step,
                    "time/train_active_hours": clock.seconds() / 3600,
                    "global_step": step,
                })
                wandb_log.log(payload, step)
            metric_sums, micro_batches, last_time, last_step, data_seconds = {}, 0, now, step, 0.0
            step_losses, step_grad_norms = [], []
        elif step % cfg.log_every == 0:
            metric_sums, micro_batches, last_time, last_step, data_seconds = {}, 0, time.time(), step, 0.0
            step_losses, step_grad_norms = [], []

        if (step % cfg.ckpt_every == 0 or step == cfg.num_steps) and is_main():
            with clock.paused():
                save_checkpoint(ckpt_dir, step, model, ema, optimizer, cfg)

        if diag_on and (step % cfg.diag_every == 0 or step == start_step + 1):
            with clock.paused():  # collective: every rank measures on its own data, then the results are averaged
                # replicates of the baseline micro-batch size: an --mlmc batch is packed (sum(mlmc_batch) samples), far too big to
                # measure the reference and every level on whole
                replicates = replicate_batches(lambda: to_device(next(batches)), cfg.diag_replicates, cfg.micro_batch_size)
                # The first measurement (step 1, untrained model, sizes from its variances) is provisional:
                # the next diagnostic re-times the costs.
                costs_due = (diag_cost_cache is None or diag_cost_step == start_step + 1
                             or cfg.diag_cost_every == 0 or step % cfg.diag_cost_every == 0)
                meas = measure(model, replicates, diag_level_weights, diag_exit_weights, loss_kwargs, device, autocast,
                               timing_sizes, ref_batch, cost_cache=None if costs_due else diag_cost_cache, base_depth=base_depth,
                               common_noise=cfg.diag_common_noise, deep_exit_weights=deep_ref_weights, max_chunk=max_chunk)
                if costs_due:
                    diag_cost_cache, diag_cost_step = meas.cost_cache(), step
                if device.type == "cuda":
                    torch.cuda.empty_cache()  # the timing forwards leave the allocator full, and NCCL needs its own buffers
                meas = reduce_mean(meas, len(diag_depths), device)
                report = build_report(meas, block_costs, block_ref, cfg.mlmc_batch if mlmc else None, world_size() * accum,
                                      depths=diag_depths, budget_batch=alloc_batch, deep_block_ref=deep_block_ref)
                report["mlmc_diag/base_depth"] = float(base_depth)
                report["mlmc_diag/final_exit_only"] = float(diag_final_only)
                report["mlmc_diag/costs_measured_this_step"] = float(costs_due)
                if costs_due and is_main():
                    for depth, asked, used in zip(diag_depths, meas.timing_wanted, meas.timing_sizes):
                        if used < asked:
                            print(f"WARNING: level {depth} was timed at {used:.0f} samples, not {asked:.0f} (out of memory, batch halved): "
                                  "its cost per sample is that of a smaller batch (mlmc_diag/level*/timing_halved)", flush=True)
                    if meas.ref_batch < ref_batch:
                        print(f"WARNING: the reference was timed at {meas.ref_batch:.0f} samples, not {ref_batch} (out of memory, batch halved)", flush=True)
            if is_main():
                print(json.dumps({"step": step, **{k: report[k] for k in report if k.startswith("mlmc_diag/") and k.count("/") == 1}}), flush=True)
                smoothed = diag_ema.update(report) if diag_ema and step != start_step + 1 else {}  # skip the untrained first point
                wandb_log.log({**report, **smoothed, "global_step": step, "time/train_active_hours": clock.seconds() / 3600}, step)

        if cfg.fid_every and step % cfg.fid_every == 0:
            with clock.paused():  # collective
                fid_images: dict[int, torch.Tensor] = {}
                fid_metrics = run_fid(ema if cfg.fid_use_ema else model, text_encoder, cfg, out_dir, step, device, dtype,
                                      use_ema=cfg.fid_use_ema, images_out=fid_images)
            if is_main():
                print(json.dumps({"step": step, **fid_metrics}), flush=True)
                payload = {**fid_metrics, "global_step": step, "time/train_active_hours": clock.seconds() / 3600}
                if wandb_on:
                    for depth, grid in fid_images.items():
                        payload[f"fid/images_loops{depth}"] = wandb_log.image(
                            grid, f"FID samples, loops {depth}, {len(grid)} images, cfg {cfg.fid_cfg_scale}, {cfg.fid_steps} steps")
                wandb_log.log(payload, step)

        if cfg.sample_every and step % cfg.sample_every == 0 and is_main():
            with clock.paused():
                samples = euler_sample(ema, text[:4], text_mask[:4], cfg.image_size, steps=25, cfg_scale=cfg.sample_cfg_scale,
                                       noise_scale=cfg.noise_scale)
                path = out_dir / "samples" / f"{step:07d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                save_image(torch.nan_to_num((samples.float() + 1) / 2), path, nrow=2)
                if wandb_on:
                    payload = {"samples/ema": wandb_log.image(samples, f"EMA samples, 25 steps, cfg {cfg.sample_cfg_scale}")}
                    payload.update(wandb_log.tensor_stats("samples/ema", samples))
                    wandb_log.log(payload, step)
        if is_main() and cfg.viz_every and step % cfg.viz_every == 0:
            with clock.paused():
                grid, stats = exit_visualization(model, images, text, text_mask, cfg, device, autocast, num_levels)
                path = out_dir / "viz" / f"{step:07d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(wandb_log.to_uint8(grid, nrow=1)).save(path)
                if wandb_on:
                    wandb_log.log({"viz/exits": wandb_log.image(grid, "x0 | x_t | loop 1 .. K", nrow=1), **stats}, step)
    wandb_log.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
