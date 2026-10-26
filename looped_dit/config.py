"""Training configuration: one flat dataclass, loaded from YAML."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Config values may reference ${VAR}. VAR is read from the environment; these
# three fall back to directories inside the repository.
DEFAULT_ROOTS = {
    "DATA_ROOT": REPO_ROOT / "data",
    "OUTPUT_ROOT": REPO_ROOT / "outputs",
    "ASSET_ROOT": REPO_ROOT / "eval_assets",
}

DEEP_SUPERVISION_WEIGHTINGS = ("final_plus_mean", "exponential", "uniform")


def expand_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} in strings; unknown variables raise."""
    if isinstance(value, str):

        def substitute(match: re.Match) -> str:
            name = match.group(1)
            if name in os.environ:
                return os.environ[name]
            if name in DEFAULT_ROOTS:
                return str(DEFAULT_ROOTS[name])
            raise KeyError(f"undefined variable ${{{name}}} in {value!r}")

        return re.sub(r"\$\{(\w+)\}", substitute, value)
    if isinstance(value, list):
        return [expand_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_vars(v) for k, v in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return expand_vars(data)


@dataclass
class TrainConfig:
    # Architecture (MiniT2I MMDiT backbone).
    image_size: int = 512
    patch_size: int = 32
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 2.6667
    pca_channels: int = 128
    text_preamble_depth: int = 2
    loop_split: list[int] = field(default_factory=lambda: [6, 5, 6])  # pre-loop / looped / post-loop blocks

    # Looped-DiT components; each can be switched off for ablations.
    num_loops: int = 4  # loop depth N at training time; 1 = no looping
    share_loop_weights: bool = True  # False: separate core blocks per pass (the "deeper" baseline)
    deep_supervision: bool = True  # flow loss on every loop exit, not only the last
    deep_supervision_weighting: str = "final_plus_mean"  # final_plus_mean | exponential | uniform
    use_xsa: bool = True  # exclusive self-attention in the looped blocks
    use_attn_gate: bool = False  # head-wise sigmoid attention gate in the looped blocks

    # Training objective. "baseline": every sample runs the full depth and, with deep_supervision,
    # all exits are trained on it. "mlmc": the same objective telescoped over loop depth (level k =
    # depth k): level 1 trains the depth-1 exit, level k >= 2 trains the difference of exits k and k-1
    # on the same noisy input. With deep_supervision it telescopes the deep_supervision_weighting objective; without it
    # the final-only objective (every level weight is 1, the telescope sums to the final exit's loss).
    train_mode: str = "baseline"  # baseline | mlmc
    mlmc_levels: int = 0  # top depth K of the telescope, 2 <= K <= num_loops; 0 = num_loops
    # Depth of the shallowest level, 1 <= base < K. The levels are the depths base..K (K - base + 1 of them, so
    # mlmc_batch needs that many entries). The base level then runs `base` loops and reads exits 1..base on the same
    # samples; the model keeps its K loops, only the cheap levels are left out.
    mlmc_base_depth: int = 1
    # Samples per level, per GPU and micro-batch (level 1 first); empty = micro_batch_size halved per
    # level. The loader then reads sum(mlmc_batch) samples per micro-batch, and batch_size counts
    # those samples (accumulation = batch_size / (sum(mlmc_batch) * world size)).
    mlmc_batch: list[int] = field(default_factory=list)
    # Level-wise micro-batching: mlmc_batch is then the samples per level, per GPU and optimizer step (what the
    # --baseline diagnostic logs as equal_cost_samples with diag_alloc_batch = batch_size / GPUs), batch_size must be
    # sum(mlmc_batch) * GPUs (no accumulation), and every level runs in the fewest equal chunks whose activations fit in
    # mlmc_max_blocks (transformer blocks x samples per forward; 0 = micro_batch_size x the baseline's blocks per sample,
    # the activation memory of a baseline micro-batch). One backward per chunk, one gradient sync per step.
    mlmc_level_batching: bool = False
    mlmc_max_blocks: int = 0

    # Frozen text encoder.
    text_encoder: str = "google/flan-t5-large"
    text_dim: int = 1024
    prompt_length: int = 256

    # Flow matching: x_t = t x + (1 - t) eps, eps ~ N(0, noise_scale^2), t ~ logit-normal.
    noise_scale: float = 2.0
    t_logit_mean: float = -0.8
    t_logit_std: float = 0.8
    label_drop_rate: float = 0.1  # prompt dropout for classifier-free guidance

    # Data: "chunks" reads pre-tokenized tensor chunks (pretraining); "webdataset"
    # mixes image/caption WebDataset sources by weight (fine-tuning).
    dataset: str = "chunks"
    chunk_dirs: list[str] = field(default_factory=list)
    webdataset_sources: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {path, weight}
    shuffle_buffer: int = 1000
    num_workers: int = 6
    prefetch_factor: int = 4

    # Optimization. Gradient accumulation = batch_size / (micro_batch_size * world size).
    batch_size: int = 1024
    micro_batch_size: int = 32
    num_steps: int = 250_000
    warmup_steps: int = 5_000
    learning_rate: float = 4e-4
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    max_grad_norm: float = 0.1
    ema_decay: float = 0.99995
    amp_dtype: str = "bf16"
    seed: int = 42

    # Weights & Biases (enabled with --wandb). wandb_env: path to a dotenv-style file with
    # WANDB_API_KEY (and optionally WANDB_ENTITY, WANDB_BASE_URL, ...); it is read before wandb.init
    # and never printed. $LOOPED_DIT_WANDB_ENV is the fallback when this is empty.
    wandb_env: str = ""
    wandb_project: str = "looped-dit"
    wandb_entity: str = ""  # empty = the account default
    wandb_mode: str = "online"  # online | offline | disabled
    viz_every: int = 5_000  # log [x0 | x_t | prediction of every loop exit] grids (0 = off)

    # MLMC / Giles diagnostics (every diag_every steps, 0 = off; works in baseline and mlmc mode):
    # gradient-level variance V_l and compute cost C_l of every level, against the baseline objective.
    diag_every: int = 1_000
    diag_replicates: int = 8  # replicate micro-batches per estimate (>= 2); the variance error falls as 1/sqrt(replicates - 1)
    diag_common_noise: bool = True  # every term of a replicate draws the same t / noise / dropout: the ratios are far less noisy
    # --baseline diagnostics time each level in the chunks level-wise MLMC micro-batching would run it in (mlmc_max_blocks),
    # so the cost per sample is the one a level-wise step pays and a big whole-step count does not run out of memory.
    # False: each level is timed at its whole count in one forward.
    diag_level_chunks: bool = True
    diag_ema_decay: float = 0.8  # decay of the log-space moving average logged as mlmc_diag_ema/* (0 = off)
    # Telescope the --baseline diagnostics evaluate: depths diag_base_depth..num_loops. The default 1 is the full
    # 4-level telescope (levels 1, 2, 3, 4). 2 starts at 2 loops (levels 2, 3, 4): its base level runs 2 loops and
    # reads exit 1 on the same samples. Clamped to num_loops - 1. --mlmc diagnostics always follow the telescope trained.
    diag_base_depth: int = 1
    # Objective the diagnostics evaluate. "deep_supervision": the weighted objective (weights from
    # deep_supervision_weighting). "final": the final exit only, so every weight is 1 and the reference is the plain
    # depth-N loss. "auto": "final" for a run trained with deep_supervision false (--baseline or --mlmc),
    # else "deep_supervision".
    diag_objective: str = "auto"
    # Costs are model forward + backward times and stable, so they are re-measured only every
    # diag_cost_every steps (0 = at every diagnostic); in between the last measured costs are reused.
    diag_cost_every: int = 5_000
    # Batch sizes the costs are timed at. Both default (0) to micro_batch_size, what a training step really runs: the
    # baseline's forward batch for the reference, and the baseline batch whose compute budget the hypothetical --mlmc
    # allocation (the equal-cost optimum, so more samples than that batch in total) spends. Set them to
    # batch_size // GPUs to use the whole step batch (an idealized level-wise micro-batching that the current training
    # code does not do).
    diag_ref_batch: int = 0
    diag_alloc_batch: int = 0

    # Periodic FID (0 = off): fid_num_images images from the EMA weights, scored against reference
    # statistics. Give fid_stats (a torch-fidelity .npz) or fid_ref_dir (real images, see
    # tools/prepare_fid_ref.py), and fid_prompts (one prompt per line).
    fid_every: int = 0
    fid_num_images: int = 5_000
    fid_prompts: str = ""
    fid_stats: str = ""
    fid_ref_dir: str = ""
    fid_loops: list[int] = field(default_factory=list)  # loop depths to score; empty = the trained depth
    fid_use_ema: bool = True  # False: score the raw training weights instead of the EMA
    fid_steps: int = 50
    fid_cfg_scale: float = 6.0
    fid_batch_size: int = 16
    fid_log_images: int = 8  # the first N generated FID images of each scored loop depth go to wandb as a grid (0 = off)

    # Logging and checkpoints.
    log_every: int = 100
    sample_every: int = 10_000
    sample_cfg_scale: float = 6.0  # guidance of the sample grids; 1.0 when there is no unconditional branch
    ckpt_every: int = 10_000
    ckpt_keep_every: int = 50_000  # checkpoints at multiples of this are never pruned
    keep_last: int = 3

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainConfig:
        cfg = cls()
        names = {f.name for f in fields(cls)}
        for key, value in values.items():
            if key not in names:
                raise KeyError(f"unknown config key: {key}")
            default = getattr(cfg, key)
            if type(default) in (int, float) and isinstance(value, str):  # YAML reads "2e-4" as a string
                value = type(default)(float(value))
            elif type(default) is float and type(value) is int:
                value = float(value)
            setattr(cfg, key, value)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
        return cls.from_dict({**load_yaml(path), **(overrides or {})})

    def validate(self) -> None:
        if len(self.loop_split) != 3:
            raise ValueError(f"loop_split must have three entries (pre, core, post), got {self.loop_split}")
        if self.deep_supervision_weighting not in DEEP_SUPERVISION_WEIGHTINGS:
            raise ValueError(f"deep_supervision_weighting must be one of {DEEP_SUPERVISION_WEIGHTINGS}")
        if self.deep_supervision and self.num_loops < 2:
            raise ValueError("deep_supervision needs num_loops >= 2 (it supervises the intermediate loops)")
        if self.train_mode not in ("baseline", "mlmc"):
            raise ValueError(f"train_mode must be 'baseline' or 'mlmc', got {self.train_mode!r}")
        if self.train_mode == "mlmc":
            levels = self.num_mlmc_levels()
            if not 2 <= levels <= self.num_loops:
                raise ValueError(f"mlmc_levels must lie in [2, num_loops={self.num_loops}], got {levels}")
            if not self.share_loop_weights and levels != self.num_loops:
                raise ValueError("without shared loop weights mlmc needs mlmc_levels == num_loops (every pass has its own blocks)")
            if not 1 <= self.mlmc_base_depth < levels:
                raise ValueError(f"mlmc_base_depth must lie in [1, mlmc_levels={levels}), got {self.mlmc_base_depth}")
            if not self.share_loop_weights and self.mlmc_base_depth != 1:
                raise ValueError("without shared loop weights mlmc needs mlmc_base_depth == 1")
            count = self.mlmc_level_count()
            if self.mlmc_batch and (len(self.mlmc_batch) != count or min(self.mlmc_batch) < 1):
                raise ValueError(f"mlmc_batch needs {count} positive entries (one per level, depths "
                                 f"{self.mlmc_base_depth}..{levels}), got {self.mlmc_batch}")
            if not self.mlmc_batch and self.micro_batch_size < count:
                raise ValueError(f"micro_batch_size {self.micro_batch_size} is below the {count} MLMC levels")
            if self.mlmc_level_batching and not self.mlmc_batch:
                raise ValueError("mlmc_level_batching needs mlmc_batch: the samples per level and step (see the config comment)")
            if self.mlmc_max_blocks < 0:
                raise ValueError("mlmc_max_blocks must be >= 0")
        elif self.mlmc_batch or self.mlmc_levels or self.mlmc_base_depth != 1 or self.mlmc_level_batching:
            raise ValueError("mlmc_levels / mlmc_base_depth / mlmc_batch / mlmc_level_batching only apply with train_mode: mlmc")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError(f"wandb_mode must be online, offline or disabled, got {self.wandb_mode!r}")
        if self.diag_objective not in ("auto", "deep_supervision", "final"):
            raise ValueError(f"diag_objective must be auto, deep_supervision or final, got {self.diag_objective!r}")
        if not 0 <= self.diag_ema_decay < 1:
            raise ValueError(f"diag_ema_decay must be in [0, 1), got {self.diag_ema_decay}")
        if self.diag_base_depth < 1:
            raise ValueError(f"diag_base_depth must be >= 1, got {self.diag_base_depth}")
        if min(self.diag_cost_every, self.diag_ref_batch, self.diag_alloc_batch) < 0:
            raise ValueError("diag_cost_every, diag_ref_batch and diag_alloc_batch must be >= 0")
        if self.diag_every and self.diag_replicates < 2:
            raise ValueError("diag_replicates must be >= 2 (a variance needs two replicates)")
        if self.fid_every:
            if not self.fid_prompts or not (self.fid_stats or self.fid_ref_dir):
                raise ValueError("fid_every needs fid_prompts and one of fid_stats / fid_ref_dir")
            if self.fid_log_images < 0:
                raise ValueError("fid_log_images must be >= 0")
            if any(r < 1 for r in self.fid_loops) or (
                not self.share_loop_weights and any(r > self.num_loops for r in self.fid_loops)
            ):
                raise ValueError(f"fid_loops must be >= 1 (and <= num_loops={self.num_loops} without shared loop weights)")
        if self.dataset not in ("chunks", "webdataset"):
            raise ValueError(f"dataset must be 'chunks' or 'webdataset', got {self.dataset!r}")
        if self.amp_dtype not in ("bf16", "fp32"):  # fp16 would need loss scaling
            raise ValueError(f"amp_dtype must be bf16 or fp32, got {self.amp_dtype!r}")

    def num_mlmc_levels(self) -> int:
        """The top depth K of the telescope (the number of levels when it starts at depth 1)."""
        return self.mlmc_levels or self.num_loops

    def mlmc_level_count(self) -> int:
        """Levels actually trained: the depths mlmc_base_depth..K."""
        return self.num_mlmc_levels() - self.mlmc_base_depth + 1

    def model_kwargs(self) -> dict[str, Any]:
        return dict(
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            mlp_ratio=self.mlp_ratio,
            pca_channels=self.pca_channels,
            text_dim=self.text_dim,
            text_preamble_depth=self.text_preamble_depth,
            loop_split=tuple(self.loop_split),
            num_loops=self.num_loops,
            share_loop_weights=self.share_loop_weights,
            use_xsa=self.use_xsa,
            use_attn_gate=self.use_attn_gate,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
