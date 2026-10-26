#!/bin/bash
#SBATCH --job-name=LoopedDiT
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=240G
#SBATCH --time=01-00:00:00
# The image's PyTorch (2.5.1 in jit.sqsh) supports <= sm_90 (Hopper): exclude Blackwell partitions.
#SBATCH --partition=H200,H200-SDS,H100,RTXB6000,H100-PCI,H200-PCI
#SBATCH --output=slurm_logs/looped_dit_%j.out
#SBATCH --error=slurm_logs/looped_dit_%j.err

# Usage (run `mkdir -p slurm_logs` once before the first sbatch):
#   sbatch run.sh /netscratch/$USER/looped_dit/experiments/baseline --baseline
#   sbatch run.sh /netscratch/$USER/looped_dit/experiments/mlmc --mlmc --mlmc-levels 4 --mlmc-batch 4 2 1 1
#   sbatch --gpus=2 run.sh OUT_DIR --mlmc ...                 # fewer GPUs (accumulation adapts)
#   CONFIG=configs/b32_pretrain.yml sbatch run.sh OUT_DIR ...
#   OUTPUT_DIR=/path/to/out sbatch run.sh --mlmc               # env fallback for the output directory
# Everything after the optional output directory goes to `python -m looped_dit.train`
# (e.g. --set fid_every=10000 fid_prompts=... ). Resubmitting with the same output directory
# resumes from its newest checkpoint and continues the same wandb run.

# ---- cluster paths -------------------------------------------------------------------
export DATA_ROOT="${DATA_ROOT:-/netscratch/$USER/Looped-DiT/data}"        # cc12m_chunks/, fluxreason_chunks/, ...
export OUTPUT_ROOT="${OUTPUT_ROOT:-/netscratch/$USER/Looped-DiT/outputs}"
export ASSET_ROOT="${ASSET_ROOT:-/netscratch/$USER/Looped-DiT/eval_assets}"
export HF_HOME="${HF_HOME:-/netscratch/$USER/hf_cache}"                  # FLAN-T5 is downloaded here once
export WANDB_ENV="${WANDB_ENV:-/netscratch/$USER/.wandb_env}"  # dotenv file: WANDB_API_KEY=...
export WANDB="${WANDB:-1}"                                               # 0 = no wandb
export CONFIG="${CONFIG:-configs/imagnet_pretrain.yml}"
# ------------------------------------------------------------------------------------------------

# Optional first positional arg = OUTPUT_DIR (must not start with '-')
if [[ $# -gt 0 && "${1}" != -* ]]; then
  export OUTPUT_DIR="${1}"
  shift
fi
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/b16_run}"

# Inherit GPU/CPU/mem/time from the #SBATCH allocation. Re-requesting --gpus /
# --partition / --mem on srun causes "Invalid generic resource (gres) specification" on some nodes.
# On Pegasus, --ntasks plus --gpus-per-task binds the task to one GPU, so these variables
# report 1. --gpus with --gpu-bind=none is what makes all four visible. The container
# recounts with nvidia-smi before torchrun.
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  export NPROC="${SLURM_GPUS_ON_NODE}"
else
  export NPROC="${NPROC:-4}"
fi

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "IMAGE=${LOOPED_DIT_IMAGE:-/fscratch/attia/container_images/loopDiT.sqsh}"
echo "NPROC=${NPROC} node=${SLURMD_NODENAME:-unknown} CONFIG=${CONFIG}"
echo "Extra args: $*"

mkdir -p slurm_logs "${OUTPUT_DIR}"

srun -u -K \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,/ds-sds:/ds-sds,/fscratch:/fscratch,"$(pwd)":"$(pwd)" \
  --container-image="${LOOPED_DIT_IMAGE:-/fscratch/attia/container_images/loopDiT.sqsh}" \
  --container-workdir="$(pwd)" \
  --export=ALL \
  bash train_looped_dit.sh "$@"
