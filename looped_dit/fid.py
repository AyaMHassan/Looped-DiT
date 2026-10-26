"""Periodic FID of the EMA model, scored with torch-fidelity.

Every rank generates its share of fid_num_images images (prompts cycle through fid_prompts) into
a temporary folder under the output directory, which must be on a filesystem shared by the ranks;
rank 0 scores the folder against the reference statistics (fid_stats, a torch-fidelity .npz) or
the real images in fid_ref_dir (features cached by torch-fidelity), and the folder is deleted.
The sampler noise is seeded identically at every evaluation, and the training RNG state is
restored afterwards, so FID does not perturb training.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from .config import TrainConfig
from .diffusion import euler_sample
from .pipeline import TextEncoder, to_pil
from .utils import is_main, rank, world_size


def load_prompts(path: str | Path) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    return prompts


def fid_kwargs(folder: Path, cfg: TrainConfig) -> dict:
    """Arguments of torch_fidelity.calculate_metrics for scoring `folder`. fid_stats (a .npz with `mu`
    and `sigma`) needs the LTH14 fork of torch-fidelity; upstream only takes a second input."""
    kwargs = dict(input1=str(folder), cuda=torch.cuda.is_available(), fid=True, verbose=False)  # the other metrics default to off
    if cfg.fid_stats:
        kwargs["fid_statistics_file"] = cfg.fid_stats
        # The fork reads the file only on its general path. With fid as the sole metric it takes a shortcut that
        # needs input2 ("Input descriptor ... field"), so a second metric keeps it off that shortcut. Inception
        # score comes from the same Inception pass, so this adds almost no time.
        kwargs["isc"] = True
    else:
        kwargs.update(input2=cfg.fid_ref_dir, input2_cache_name=ref_cache_name(cfg))
    return kwargs


def check_fid_setup(cfg: TrainConfig) -> None:
    """Fail at startup, not after the first fid_every steps."""
    if not Path(cfg.fid_prompts).is_file():
        raise FileNotFoundError(f"fid_prompts not found: {cfg.fid_prompts}")
    if cfg.fid_stats and not Path(cfg.fid_stats).is_file():
        raise FileNotFoundError(f"fid_stats not found: {cfg.fid_stats}")
    if not cfg.fid_stats and not Path(cfg.fid_ref_dir).is_dir():
        raise FileNotFoundError(f"fid_ref_dir not found: {cfg.fid_ref_dir}")
    import torch_fidelity  # noqa: F401  (pip install torch-fidelity)
    from torch_fidelity.defaults import DEFAULTS

    # calculate_metrics silently ignores options it does not know, which would disable the reference
    # cache (or the statistics file) without any error: check every name against the installed version.
    unknown = [key for key in fid_kwargs(Path("."), cfg) if key not in DEFAULTS]
    if unknown:
        raise RuntimeError(
            f"the installed torch-fidelity does not know the option(s) {unknown}"
            + (" (fid_stats needs the LTH14 fork: pip install git+https://github.com/LTH14/torch-fidelity.git)" if "fid_statistics_file" in unknown else "")
        )


def ref_cache_name(cfg: TrainConfig) -> str:
    """torch-fidelity cache key of the reference features: unique per folder (two folders both
    called `images` must not share features) and per resolution."""
    digest = hashlib.sha1(str(Path(cfg.fid_ref_dir).resolve()).encode()).hexdigest()[:12]
    return f"looped_dit_{digest}_{cfg.image_size}px"


def score_folder(folder: Path, cfg: TrainConfig) -> dict[str, float]:
    import torch_fidelity

    return torch_fidelity.calculate_metrics(**fid_kwargs(folder, cfg))


@torch.no_grad()
def run_fid(
    ema: torch.nn.Module,
    text_encoder: TextEncoder,
    cfg: TrainConfig,
    out_dir: Path,
    step: int,
    device: torch.device,
    dtype: torch.dtype,
    use_ema: bool = True,
    images_out: dict[int, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Collective: call on every rank. Returns the metrics on rank 0 and {} elsewhere. With images_out and
    cfg.fid_log_images, rank 0 fills it with the first fid_log_images generated images per scored loop depth
    (NCHW in [-1, 1], on the CPU) before the folders are deleted.
    `ema` is whichever module is scored (the EMA copy, or the training model with fid_use_ema false);
    its train/eval mode is restored afterwards. use_ema is only recorded in the metrics."""
    start = time.time()
    prompts = load_prompts(cfg.fid_prompts)
    loops = list(cfg.fid_loops) or [cfg.num_loops]
    root = out_dir / "fid_tmp" / f"{step:07d}"
    ranks, me = world_size(), rank()
    total = cfg.fid_num_images
    mine = list(range(me, total, ranks))  # global image indices of this rank

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    was_training = ema.training
    ema.eval()
    try:
        for depth in loops:
            folder = root / f"loops{depth}"
            folder.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(cfg.seed + 7919 + me)  # the same noise at every evaluation
            for i in range(0, len(mine), cfg.fid_batch_size):
                idx = mine[i : i + cfg.fid_batch_size]
                ids, mask = text_encoder.tokenize([prompts[j % len(prompts)] for j in idx])
                with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                    text = text_encoder.encode(ids, mask).to(dtype)
                images = euler_sample(ema, text, mask, cfg.image_size, cfg.fid_steps, cfg.fid_cfg_scale,
                                      cfg.noise_scale, depth)
                for j, im in zip(idx, to_pil(images)):
                    im.save(folder / f"{j:07d}.png")
    finally:
        ema.train(was_training)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
    if dist.is_initialized():
        dist.barrier()
    generated = time.time() - start

    metrics: dict[str, float] = {}
    if is_main():
        for depth in loops:
            result = score_folder(root / f"loops{depth}", cfg)
            metrics[f"fid/fid_loops{depth}"] = float(result["frechet_inception_distance"])
            if images_out is not None and cfg.fid_log_images:
                files = sorted((root / f"loops{depth}").glob("*.png"))[: cfg.fid_log_images]
                images_out[depth] = torch.stack(
                    [torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))).permute(2, 0, 1).float() / 127.5 - 1 for f in files])
        metrics["fid/fid"] = metrics[f"fid/fid_loops{loops[-1]}"]  # the deepest scored loop count
        metrics.update({"fid/use_ema": float(use_ema), "fid/num_images": float(total), "fid/steps": float(cfg.fid_steps), "fid/cfg_scale": cfg.fid_cfg_scale,
                        "fid/generation_seconds": generated, "fid/total_seconds": time.time() - start})
        shutil.rmtree(root, ignore_errors=True)
    if dist.is_initialized():
        dist.barrier()
    return metrics
