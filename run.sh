#!/bin/bash
#SBATCH --job-name=Poly-Attention
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:45:00
#SBATCH --output=out/%x_%j.log

set -euo pipefail

cd /home/viopazo/PolyAttention

mkdir -p out datasets experiments/runs

source /home/viopazo/venvs/poly-attention/bin/activate

export PYTHONUNBUFFERED=1

SCRIPT=${1:?Usage: sbatch run.sh experiments/scripts/<script.py> [args...]}

shift || true

echo "Project dir: /home/viopazo/PolyAttention"
echo "Venv:        /home/viopazo/venvs/poly-attention"
echo "Python:      $(which python)"
echo "Script:      $SCRIPT"
echo "Args:        $*"
echo "Job id:      ${SLURM_JOB_ID:-manual}"
echo

python -u "$SCRIPT" "$@"
