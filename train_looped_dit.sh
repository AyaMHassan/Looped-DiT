#!/bin/bash
# Launch Looped-DiT training (called from run.sh inside the container; also usable directly:
#   OUTPUT_DIR=outputs/test NPROC=1 bash train_looped_dit.sh --baseline --wandb-mode disabled).
set -euo pipefail

# conda activate scripts may reference unset variables; nounset breaks them.
set +u
source /root/miniconda3/etc/profile.d/conda.sh
# The new image already has requirements.txt installed. Activate jit/ssd only when this
# Python does not have them, and never prefer an old .python_deps tree over the image.
if [[ -n "${PYTHONPATH:-}" ]]; then
  _pp=""
  IFS=':' read -ra _parts <<< "${PYTHONPATH}"
  for _p in "${_parts[@]}"; do
    [[ "${_p}" == *"/.python_deps/"* ]] && continue
    _pp="${_pp:+${_pp}:}${_p}"
  done
  export PYTHONPATH="${_pp}"
fi
if ! python -c "import transformers" >/dev/null 2>&1; then
  for env in "${CONDA_ENV:-loopDiT}" loopDiT; do
    if conda env list | grep -qE "^${env}\s"; then conda activate "${env}"; break; fi
  done
fi
set -u

# Repo root = directory containing this script (stable even if sbatch's cwd differs).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

python -c "
import torch, transformers, webdataset, yaml, wandb
from transformers import T5EncoderModel
from torchao.prototype.safetensors.safetensors_utils import is_metadata_torchao
from torch_fidelity.defaults import DEFAULTS
assert 'fid_statistics_file' in DEFAULTS
print('torch', torch.__version__, '| transformers', transformers.__version__, '| python', __import__('sys').executable)
" || { echo "A dependency is still missing in this container. pip install -r requirements.txt" >&2; exit 1; }

export TOKENIZERS_PARALLELISM=false
: "${OUTPUT_DIR:?set OUTPUT_DIR (run.sh does)}"
# Count devices the container can actually see. A Pegasus --gpus-per-task bind leaves
# CUDA_VISIBLE_DEVICES as a single index even when more GPUs were requested.
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  _listed="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  if [[ "${_listed}" -gt 0 ]]; then NPROC="${_listed}"; fi
fi
: "${NPROC:=1}"
echo "visible GPUs: NPROC=${NPROC} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
: "${CONFIG:=configs/b16_pretrain.yml}"

# Default config and wandb flags unless the caller passed their own.
args=("$@")
has() { local a; for a in ${args[@]+"${args[@]}"}; do [[ "$a" == "$1" ]] && return 0; done; return 1; }
if ! has --config; then args=(--config "${CONFIG}" ${args[@]+"${args[@]}"}); fi
if [[ "${WANDB:-1}" != "0" ]]; then
  has --wandb || args+=(--wandb)
  if [[ -n "${WANDB_ENV:-}" ]] && ! has --wandb-env; then
    if [[ -f "${WANDB_ENV}" ]]; then args+=(--wandb-env "${WANDB_ENV}")
    else echo "WARNING: WANDB_ENV=${WANDB_ENV} not found; relying on an existing wandb login / WANDB_API_KEY" >&2; fi
  fi
fi
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
echo "torchrun --nproc_per_node=${NPROC} -m looped_dit.train --output-dir ${OUTPUT_DIR} ${args[*]}"
exec torchrun --standalone --nproc_per_node="${NPROC}" -m looped_dit.train --output-dir "${OUTPUT_DIR}" "${args[@]}"
