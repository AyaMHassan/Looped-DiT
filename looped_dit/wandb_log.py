"""Weights & Biases helpers: credentials from an env file, rank-0 logging, images, active time.

The env file is dotenv-style (`KEY=VALUE` or `export KEY=VALUE`, `#` comments), typically holding
WANDB_API_KEY and optionally WANDB_ENTITY / WANDB_PROJECT / WANDB_BASE_URL. Values are put in
os.environ unless the variable is already set, and are never printed.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from .config import TrainConfig

_run = None


def load_env_file(path: str | Path) -> list[str]:
    """Export the variables of a dotenv-style file; returns the names that were set."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"wandb env file not found: {path}")
    names = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            names.append(key)
    return names


def _id_was_deleted(exc: BaseException) -> bool:
    """W&B refuses to recreate a run id that was deleted in the UI."""
    return "previously created and deleted" in str(exc)


def init_wandb(cfg: TrainConfig, out_dir: Path, run_name: str, config: dict[str, Any], env_file: str = "") -> bool:
    """Start the run (call on rank 0 only). Returns whether a run is active.

    The run id is the output-directory name, so a restarted job resumes the same W&B run.
    A deleted id cannot be reused; that case starts a new run and remembers its id in
    ``wandb_run_id`` next to the checkpoints.
    """
    global _run
    env_file = env_file or cfg.wandb_env or os.environ.get("LOOPED_DIT_WANDB_ENV", "")
    if env_file:
        names = load_env_file(env_file)
        print(f"wandb: loaded {len(names)} variable(s) from {env_file}", flush=True)
    if cfg.wandb_mode == "disabled":
        return False
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without it", flush=True)
        return False
    os.environ["WANDB_MODE"] = cfg.wandb_mode  # the config decides, not a stale value in an env file
    project = cfg.wandb_project or os.environ.get("WANDB_PROJECT", "looped-dit")
    entity = cfg.wandb_entity or os.environ.get("WANDB_ENTITY") or None
    out_dir.mkdir(parents=True, exist_ok=True)
    id_path = out_dir / "wandb_run_id"
    saved = id_path.read_text().strip() if id_path.is_file() else ""
    run_id = saved or run_name

    def start(run_id: str):
        return wandb.init(project=project, entity=entity, name=run_name, id=run_id, resume="allow",
                          dir=str(out_dir), config=config, mode=cfg.wandb_mode)

    try:
        _run = start(run_id)
    except Exception as exc:  # auth or network problems must not kill a training run
        if not _id_was_deleted(exc):
            print(f"wandb init failed ({exc}); continuing without it", flush=True)
            _run = None
            return False
        run_id = f"{run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"wandb: id {saved or run_name!r} was deleted; starting a new run {run_id!r}", flush=True)
        try:
            _run = start(run_id)
        except Exception as exc2:
            print(f"wandb init failed ({exc2}); continuing without it", flush=True)
            _run = None
            return False
    if run_id != run_name:
        id_path.write_text(run_id + "\n")
    wandb.define_metric("global_step")
    wandb.define_metric("time/train_active_hours")
    wandb.define_metric("compute/cum_block_samples")
    for prefix in ("train/*", "mlmc/*", "mlmc_diag/*", "mlmc_loss_proxy/*", "timing/*", "runtime/*", "compute/*"):
        wandb.define_metric(prefix, step_metric="global_step")
    wandb.define_metric("fid/*", step_metric="global_step")
    print(f"wandb: run {run_name!r} in project {project!r} (entity {entity or 'default'}, mode {cfg.wandb_mode})", flush=True)
    return True


def active() -> bool:
    return _run is not None


def log(payload: dict[str, Any], step: int) -> None:
    if _run is None or not payload:
        return
    try:
        import wandb

        wandb.log(payload, step=step)
    except Exception as exc:
        print(f"wandb log failed: {exc}", flush=True)


def to_uint8(grid: torch.Tensor, nrow: int | None = None):
    """A [-1, 1] CHW (or NCHW, tiled with nrow images per row, default square-ish) tensor as a uint8
    HWC array. Non-finite values become mid-gray pixels instead of poisoning the whole picture, and
    nothing is rescaled: gray stays gray and black stays black."""
    from torchvision.utils import make_grid

    x = grid.detach().float().cpu()
    if x.dim() == 4:
        x = make_grid(x, nrow=nrow or int(x.shape[0] ** 0.5 + 0.999))
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
    return ((x + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()


def image(grid: torch.Tensor, caption: str = "", nrow: int | None = None):
    """A [-1, 1] tensor as a wandb.Image (converted here, so wandb's own rescaling cannot change it)."""
    import wandb
    from PIL import Image

    return wandb.Image(Image.fromarray(to_uint8(grid, nrow)), caption=caption)


def tensor_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    """Scalars that tell a black picture from NaNs, a constant and a real image."""
    x = x.detach().float().cpu()
    finite = torch.isfinite(x)
    f = x[finite]
    return {
        f"{prefix}_nonfinite": float((~finite).sum()),
        f"{prefix}_mean": float(f.mean()) if f.numel() else float("nan"),
        f"{prefix}_std": float(f.std()) if f.numel() > 1 else float("nan"),
        f"{prefix}_min": float(f.min()) if f.numel() else float("nan"),
        f"{prefix}_max": float(f.max()) if f.numel() else float("nan"),
    }


def finish() -> None:
    global _run
    if _run is None:
        return
    try:
        import wandb

        wandb.finish()
    except Exception:
        pass
    _run = None


class ActiveClock:
    """Wall-clock of training proper: diagnostics, FID, sampling and checkpointing are paused out,
    so runs with different amounts of monitoring stay comparable on a time axis."""

    def __init__(self, start_seconds: float = 0.0):
        self.total, self._since = start_seconds, time.perf_counter()

    def seconds(self) -> float:
        return self.total + (time.perf_counter() - self._since if self._since is not None else 0.0)

    @contextmanager
    def paused(self):
        self.total, self._since = self.seconds(), None
        try:
            yield
        finally:
            self._since = time.perf_counter()
