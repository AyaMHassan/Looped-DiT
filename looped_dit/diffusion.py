"""Flow-matching objective with deep supervision, and the Euler sampler.

The denoiser predicts the clean image x0; the loss compares the implied velocity
(x0_hat - x_t) / (1 - t) with the true one, clamping 1 - t at 0.05 so the weight
stays bounded near t = 1.

Two training objectives share the loss definition: the baseline (training_loss: deep
supervision on one full-depth pass per sample) and multilevel Monte Carlo over loop
depth (mlmc_training_loss: the same objective, telescoped over exits).
"""

from __future__ import annotations

import torch
from torch import nn

# Exit weights are rescaled to sum to this, so the weightings differ only in shape
# (the gradient-clipping threshold is absolute, so the total scale matters).
WEIGHT_SUM = 2.0
# Exponential weighting: each exit weighs this much of the one after it.
EXPONENTIAL_RATIO = 0.5


def deep_supervision_weights(num_loops: int, weighting: str) -> list[float]:
    """Loss weights (w_1, ..., w_N) of the exits after each loop; w_N is the final one.

    At N = 4 (before rescaling to WEIGHT_SUM):
        uniform          1,   1,   1,   1
        final_plus_mean  1/3, 1/3, 1/3, 1
        exponential      1/8, 1/4, 1/2, 1
    """
    if num_loops < 2:
        raise ValueError("deep supervision needs at least two loops")
    if weighting == "uniform":
        raw = [1.0] * num_loops
    elif weighting == "final_plus_mean":
        raw = [1.0 / (num_loops - 1)] * (num_loops - 1) + [1.0]
    elif weighting == "exponential":
        raw = [EXPONENTIAL_RATIO ** (num_loops - r) for r in range(1, num_loops + 1)]
    else:
        raise ValueError(f"unknown deep supervision weighting: {weighting!r}")
    scale = WEIGHT_SUM / sum(raw)
    return [w * scale for w in raw]


def corrupt(
    images: torch.Tensor,
    text_mask: torch.Tensor,
    noise_scale: float,
    t_logit_mean: float,
    t_logit_std: float,
    label_drop_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw (label drop, t, noise) and return the noisy images x_t, the velocity
    scale 1 - t (clamped), the true velocity target and the (possibly dropped) text mask."""
    b = images.shape[0]
    if label_drop_rate > 0:
        drop = torch.rand(b, device=images.device) < label_drop_rate
        text_mask = torch.where(drop[:, None], torch.zeros_like(text_mask), text_mask)
    t = torch.sigmoid(torch.randn(b, device=images.device) * t_logit_std + t_logit_mean).clamp(1e-5, 1 - 1e-5)
    t = t[:, None, None, None]
    noise = torch.randn_like(images) * noise_scale
    x_t = images * t + noise * (1.0 - t)
    velocity_scale = (1.0 - t).clamp_min(0.05)
    target = (images - x_t) / velocity_scale
    return x_t, velocity_scale, target, text_mask


def per_sample_flow_loss(
    pred_x0: torch.Tensor, x_t: torch.Tensor, velocity_scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Flow loss of an x0 prediction, one value per sample: [B]."""
    return ((pred_x0 - x_t) / velocity_scale - target).pow(2).mean(dim=(1, 2, 3))


def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    exit_weights: list[float] | None = None,
    noise_scale: float = 2.0,
    t_logit_mean: float = -0.8,
    t_logit_std: float = 0.8,
    label_drop_rate: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Flow-matching loss. With exit_weights (one per loop, see
    deep_supervision_weights) every loop exit is decoded and trained on the same
    target (deep supervision); without, only the final prediction is."""
    x_t, velocity_scale, target, text_mask = corrupt(
        images, text_mask, noise_scale, t_logit_mean, t_logit_std, label_drop_rate
    )

    def flow_loss(pred_x0: torch.Tensor) -> torch.Tensor:
        return per_sample_flow_loss(pred_x0, x_t, velocity_scale, target).mean()

    if exit_weights is None:
        loss = flow_loss(model(x_t, text, text_mask))
        return loss, {"loss": loss.detach()}

    num_loops = len(exit_weights)
    final, exits = model(x_t, text, text_mask, exit_loops=tuple(range(1, num_loops)))
    loss_final = flow_loss(final)
    exit_losses = [flow_loss(exits[r]) for r in range(1, num_loops)]
    loss = exit_weights[-1] * loss_final + torch.stack([w * l for w, l in zip(exit_weights, exit_losses)]).sum()
    metrics = {"loss": loss.detach(), "loss_final": loss_final.detach()}
    metrics.update({f"loss_exit{r}": l.detach() for r, l in enumerate(exit_losses, start=1)})
    return loss, metrics


# ---------------------------------------------------------------------------
# Multilevel Monte Carlo over loop depth
# ---------------------------------------------------------------------------


def mlmc_level_weights(num_levels: int, weighting: str) -> list[float]:
    """Correction weights (W_1, ..., W_K) of the telescoped deep-supervision objective.

    With exit weights w_r, sum_r w_r F_r = W_1 F_1 + sum_{k>=2} W_k (F_k - F_{k-1})
    where W_k = sum_{r>=k} w_r is the suffix sum. The coefficient left on F_r is
    W_r - W_{r+1} = w_r, so every exit keeps its own training signal. All weights equal
    to 1 would train the final exit only.
    """
    w = deep_supervision_weights(num_levels, weighting)
    return [sum(w[k:]) for k in range(num_levels)]


def allocate_by_fractions(total: int, shares: list[float]) -> list[int]:
    """Split `total` samples over the levels in proportion to `shares`, summing to `total` exactly, with at
    least one sample per level (largest-remainder rounding)."""
    num_levels = len(shares)
    if total < num_levels:
        raise ValueError(f"need at least one sample per level: {total} samples for {num_levels} levels")
    exact = [total * share / sum(shares) for share in shares]
    sizes = [max(1, int(e)) for e in exact]
    while sum(sizes) < total:  # give the leftover samples to the largest fractional parts
        sizes[max(range(num_levels), key=lambda k: exact[k] - sizes[k])] += 1
    while sum(sizes) > total:  # the minimum of one sample per level over-allocated: trim the largest level
        sizes[max(range(num_levels), key=lambda k: sizes[k])] -= 1
    return sizes


def final_exit_weights(num_levels: int) -> tuple[list[float], list[float]]:
    """(exit weights, level weights) of the objective without deep supervision: only the final exit has weight, so
    every telescope weight W_k is 1 and the telescope F_1 + (F_2 - F_1) + ... + (F_K - F_{K-1}) is just F_K."""
    return [0.0] * (num_levels - 1) + [1.0], [1.0] * num_levels


def default_level_allocation(total: int, num_levels: int) -> list[int]:
    """Split `total` samples over the levels, halving per level (level 1 gets the most),
    with at least one sample per level."""
    return allocate_by_fractions(total, [2.0 ** (num_levels - k) for k in range(1, num_levels + 1)])


def mlmc_level_term(
    model: nn.Module,
    level: int,
    weight: float,
    images: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    noise_scale: float = 2.0,
    t_logit_mean: float = -0.8,
    t_logit_std: float = 0.8,
    label_drop_rate: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One MLMC level on its own samples: (weighted term, per-sample fine loss F_k,
    per-sample shallow loss S_{k-1} or None at level 1).

    Level 1 is W_1 mean(F_1) from a depth-1 pass. Level k >= 2 runs one pass to depth k and
    reads exits k and k-1 on the same (x_t, t, text, label drop): W_k mean(F_k - S_{k-1}).
    Both exits keep their gradients. Every call draws its own t, noise and label drop.
    """
    x_t, velocity_scale, target, text_mask = corrupt(
        images, text_mask, noise_scale, t_logit_mean, t_logit_std, label_drop_rate
    )
    if level == 1:
        fine = per_sample_flow_loss(model(x_t, text, text_mask, num_loops=1), x_t, velocity_scale, target)
        return weight * fine.mean(), fine, None
    pred, exits = model(x_t, text, text_mask, num_loops=level, exit_loops=(level - 1,))
    fine = per_sample_flow_loss(pred, x_t, velocity_scale, target)
    shallow = per_sample_flow_loss(exits[level - 1], x_t, velocity_scale, target)
    return weight * (fine - shallow).mean(), fine, shallow


def mlmc_base_level_term(
    model: nn.Module,
    depth: int,
    exit_weights: list[float],
    level_weights: list[float],
    images: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    noise_scale: float = 2.0,
    t_logit_mean: float = -0.8,
    t_logit_std: float = 0.8,
    label_drop_rate: float = 0.1,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Base level of a telescope that starts at `depth` loops instead of 1: (term, per-sample loss of every exit read,
    keyed by loop; the base depth itself is always included).

    One pass to `depth` loops reads exits 1..depth on the same samples, and the term is
    mean( sum_{r<depth} w_r F_r + W_depth F_depth ), reading only the exits with a nonzero weight. Together with the corrections W_k (F_k - S_{k-1}) for
    k > depth this is an unbiased estimate of the same objective sum_r w_r F_r, so every exit keeps its
    weight and only the cheapest level changes: its loss is evaluated on the base samples at the cost of
    one decode per extra exit, not on a separate batch. For depth 1 this is the usual level-1 term.
    """
    x_t, velocity_scale, target, text_mask = corrupt(
        images, text_mask, noise_scale, t_logit_mean, t_logit_std, label_drop_rate
    )
    if depth == 1:
        fine = per_sample_flow_loss(model(x_t, text, text_mask, num_loops=1), x_t, velocity_scale, target)
        return level_weights[0] * fine.mean(), {1: fine}
    read = tuple(r for r in range(1, depth) if exit_weights[r - 1] != 0)  # exits without weight are not even decoded
    if read:
        pred, exits = model(x_t, text, text_mask, num_loops=depth, exit_loops=read)
    else:  # without exit_loops the model returns the prediction alone, not a (prediction, exits) pair
        pred, exits = model(x_t, text, text_mask, num_loops=depth), {}
    fine = per_sample_flow_loss(pred, x_t, velocity_scale, target)
    losses = {depth: fine}
    total = level_weights[depth - 1] * fine
    for r in read:
        losses[r] = per_sample_flow_loss(exits[r], x_t, velocity_scale, target)
        total = total + exit_weights[r - 1] * losses[r]
    return total.mean(), losses


def mlmc_base_term(*args, **kwargs) -> torch.Tensor:
    """The base level's term alone (see mlmc_base_level_term), as the diagnostics' per-replicate loss."""
    return mlmc_base_level_term(*args, **kwargs)[0]


def mlmc_training_loss(
    model: nn.Module,
    images: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    level_sizes: list[int],
    level_weights: list[float],
    base_depth: int = 1,
    exit_weights: list[float] | None = None,
    **loss_kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Telescoped deep-supervision loss: an unbiased estimate of sum_r w_r E[F_r].

    The batch is split into independent sub-batches, level_sizes[i] samples for the level at depth base_depth + i;
    the loss is the sum of the level terms (see mlmc_level_term). The correction pushes the
    fine exit down and the shallow one up; in expectation every exit keeps weight w_r.
    level_weights are the W_k of every depth 1..K (K = top depth). With base_depth > 1 the base level runs
    base_depth loops and also reads the shallower exits that have weight (mlmc_base_level_term), which needs
    exit_weights (w_1..w_K); the levels below the base depth are not run as levels of their own.
    """
    num_levels = len(level_sizes)
    top = base_depth - 1 + num_levels
    if len(level_weights) != top or sum(level_sizes) != images.shape[0]:
        raise ValueError(f"level_sizes {level_sizes} must match the batch ({images.shape[0]}) and level_weights")
    if base_depth > 1 and (exit_weights is None or len(exit_weights) != top):
        raise ValueError(f"base_depth {base_depth} needs the {top} exit weights")
    loss = images.new_zeros(())
    metrics: dict[str, torch.Tensor] = {}
    for i, (img, txt, msk) in enumerate(
        zip(images.split(level_sizes), text.split(level_sizes), text_mask.split(level_sizes))
    ):
        term, level_metrics = mlmc_level_step(
            model, base_depth + i, base_depth, level_weights, exit_weights, img, txt, msk, **loss_kwargs
        )
        loss = loss + term
        metrics.update(level_metrics)
    metrics["loss"] = loss.detach()
    return loss, metrics


def mlmc_level_step(
    model: nn.Module,
    depth: int,
    base_depth: int,
    level_weights: list[float],
    exit_weights: list[float] | None,
    images: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    **loss_kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One level (depth) on its own samples: (its weighted term, the metrics of that level, without "loss")."""
    top = len(level_weights)
    metrics: dict[str, torch.Tensor] = {}
    shallow = None
    if depth == base_depth and base_depth > 1:
        term, exit_losses = mlmc_base_level_term(model, depth, exit_weights, level_weights, images, text, text_mask, **loss_kwargs)
        fine = exit_losses[depth]
        for r, value in exit_losses.items():  # exits read on the base samples, named as in the baseline's log
            if r != depth:
                metrics[f"loss_exit{r}"] = value.mean().detach()
    else:
        term, fine, shallow = mlmc_level_term(model, depth, level_weights[depth - 1], images, text, text_mask, **loss_kwargs)
    prefix = f"mlmc/level{depth}"
    metrics[f"{prefix}/fine"] = fine.mean().detach()  # F_k, estimated on level k's own samples
    metrics[f"{prefix}/term"] = term.detach()  # this level's weighted contribution to the loss
    if shallow is not None:
        metrics[f"{prefix}/shallow"] = shallow.mean().detach()  # S_{k-1}
        metrics[f"{prefix}/correction"] = (fine - shallow).mean().detach()
    # Named as in the baseline's log.
    metrics["loss_final" if depth == top else f"loss_exit{depth}"] = fine.mean().detach()
    return term, metrics


def split_evenly(total: int, max_size: int) -> list[int]:
    """`total` split into the fewest parts of at most max_size, as equal as possible (225, 75 -> 75 75 75; 110, 57 -> 55 55)."""
    parts = -(-total // max(1, max_size))
    return [total // parts + (i < total % parts) for i in range(parts)]


def mlmc_chunk_plan(level_sizes: list[int], max_chunk: list[int]) -> list[tuple[int, int, int, float]]:
    """Level-wise micro-batching of a packed batch: (level index, start, stop, fraction) per forward, levels in order.

    Level i owns level_sizes[i] consecutive samples of the packed batch and runs them in the fewest chunks of at most
    max_chunk[i]; fraction = chunk size / level size, the factor that turns the chunks' means into the level's mean."""
    plan, start = [], 0
    for i, (n, cap) in enumerate(zip(level_sizes, max_chunk)):
        for size in split_evenly(n, cap):
            plan.append((i, start, start + size, size / n))
            start += size
    return plan


def mlmc_chunk_loss(
    model: nn.Module,
    depth: int,
    base_depth: int,
    level_weights: list[float],
    exit_weights: list[float] | None,
    images: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    fraction: float,
    **loss_kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """A chunk of level `depth`, scaled by fraction (chunk / level size): the chunks of a level sum to its full term and
    metrics, so summing the metrics of every chunk of a step gives what mlmc_training_loss reports for the packed batch."""
    term, metrics = mlmc_level_step(model, depth, base_depth, level_weights, exit_weights, images, text, text_mask, **loss_kwargs)
    loss = term * fraction
    metrics = {key: value * fraction for key, value in metrics.items()}
    metrics["loss"] = loss.detach()
    return loss, metrics


class MLMCObjective(nn.Module):
    """The MLMC loss as one module, so DDP sees a single forward per micro-batch
    (DDP cannot take several forward calls before one backward)."""

    def __init__(self, model: nn.Module, level_sizes: list[int], level_weights: list[float], base_depth: int = 1,
                 exit_weights: list[float] | None = None, **loss_kwargs):
        super().__init__()
        self.model, self.level_sizes, self.level_weights, self.loss_kwargs = model, level_sizes, level_weights, loss_kwargs
        self.base_depth, self.exit_weights = base_depth, exit_weights

    def forward(self, images: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor):
        return mlmc_training_loss(
            self.model, images, text, text_mask, self.level_sizes, self.level_weights, self.base_depth,
            self.exit_weights, **self.loss_kwargs
        )


class MLMCChunkObjective(nn.Module):
    """One chunk of one level as a module (mlmc_chunk_loss), for level-wise micro-batching: one forward, one backward,
    wrapped by DDP like MLMCObjective. The step runs the chunks under no_sync except the last, which must be the deepest
    level's (it touches every parameter, so DDP's reduction completes)."""

    def __init__(self, model: nn.Module, level_weights: list[float], base_depth: int = 1,
                 exit_weights: list[float] | None = None, **loss_kwargs):
        super().__init__()
        self.model, self.level_weights, self.base_depth, self.exit_weights = model, level_weights, base_depth, exit_weights
        self.loss_kwargs = loss_kwargs

    def forward(self, images: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor, depth: int, fraction: float):
        return mlmc_chunk_loss(self.model, depth, self.base_depth, self.level_weights, self.exit_weights,
                               images, text, text_mask, fraction, **self.loss_kwargs)


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    image_size: int,
    steps: int = 100,
    cfg_scale: float = 6.0,
    noise_scale: float = 2.0,
    num_loops: int | None = None,
) -> torch.Tensor:
    """Euler integration from t = 0 (noise) to t = 1 (image) with classifier-free
    guidance; num_loops sets the loop depth (default: the trained depth)."""
    was_training = model.training
    model.eval()
    b = text.shape[0]
    device = text.device
    x = torch.randn(b, 3, image_size, image_size, device=device, dtype=text.dtype) * noise_scale
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    null_mask = torch.zeros_like(text_mask)
    amp_dtype = text.dtype if text.dtype in (torch.float16, torch.bfloat16) else torch.float32
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        t = torch.full((b,), float(t0), device=device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=x.is_cuda and amp_dtype != torch.float32):
            pred = model(x, text, text_mask, num_loops=num_loops)
            if cfg_scale != 1.0:
                pred_uncond = model(x, text, null_mask, num_loops=num_loops)
                pred = pred_uncond + (pred - pred_uncond) * cfg_scale
        v = (pred - x) / (1.0 - t[:, None, None, None]).clamp_min(0.05)
        x = x + v * (t1 - t0)
    model.train(was_training)
    return x.clamp(-1, 1)
