#!/bin/bash
#SBATCH --job-name=ImageNetPrep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1
#SBATCH --partition=H200,H200-SDS,H100,H100-PCI,A100-80GB,RTXB6000
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/imagenet_%j.out
#SBATCH --error=slurm_logs/imagenet_%j.err

# Write ImageNet training chunks and FID statistics (mu/sigma npz).
# The first argument is the ImageNet directory (the root, or its train/ folder).
#   mkdir -p slurm_logs
#   sbatch prepare_imagenet.sh /path/to/imagenet
# IMAGE_SIZE defaults to 256. Resubmitting continues from progress.json and
# skips statistics that are already newer than the chunks.
# Chunks are uint8 (about 0.8 GB per 1024 images at 256). The full train split
# is about 240 GB at 256. Inception weights go to TORCH_HOME.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch prepare_imagenet.sh /path/to/imagenet" >&2
  exit 1
fi

export DATA_ROOT="${DATA_ROOT:-/netscratch/$USER/Looped-DiT/data}"
export HF_HOME="${HF_HOME:-/netscratch/$USER/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-/netscratch/$USER/torch_cache}"
export IMAGE_SIZE="${IMAGE_SIZE:-256}"
IMAGES="$(cd "$1" && pwd)"
export CHUNK_DIR="${DATA_ROOT}/imagenet${IMAGE_SIZE}_chunks"
export FID_DIR="${DATA_ROOT}/imagenet${IMAGE_SIZE}_fid"

mkdir -p slurm_logs "${DATA_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${FID_DIR}"

MOUNTS="/netscratch:/netscratch,/ds:/ds,/ds-sds:/ds-sds,/fscratch:/fscratch,${PWD}:${PWD}"
case "${IMAGES}" in
  /netscratch/*|/ds/*|/ds-sds/*|/fscratch/*|"${PWD}"/*) ;;
  *) MOUNTS="${MOUNTS},${IMAGES}:${IMAGES}" ;;
esac

srun -u -K \
  --container-mounts="${MOUNTS}" \
  --container-image="${LOOPED_DIT_IMAGE:-/fscratch/attia/container_images/loopDiT.sqsh}" \
  --container-workdir="$(pwd)" \
  --export=ALL \
  bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate "${CONDA_ENV:-loopDiT}" && \
    set -euo pipefail && \
    python -u tools/prepare_imagenet.py \
      --images "'"${IMAGES}"'" \
      --out "'"${CHUNK_DIR}"'" \
      --fid-stats "'"${FID_DIR}"'/stats.npz" \
      --image-size "'"${IMAGE_SIZE}"'" \
      --workers "'"${SLURM_CPUS_PER_TASK:-16}"'"'
