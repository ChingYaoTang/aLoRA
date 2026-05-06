#!/usr/bin/bash --login
#SBATCH --job-name=alora_experiments
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=0-08:00:00

# Submit to the Iris GPU partition.
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --constraint=volta32

set -euo pipefail

print_error_and_exit() {
    echo "***ERROR*** $*"
    exit 1
}

cd "${SLURM_SUBMIT_DIR}" || print_error_and_exit "Could not cd to ${SLURM_SUBMIT_DIR}"

log_dir="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "${log_dir}"

exec > "${log_dir}/${SLURM_JOB_NAME}.out"
exec 2> "${log_dir}/${SLURM_JOB_NAME}.err"

echo "Log directory: ${log_dir}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"

module purge || print_error_and_exit "No 'module' command available"
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
module load numlib/cuDNN/9.5.0.50-CUDA-12.6.0

bash run_experiments.sh
