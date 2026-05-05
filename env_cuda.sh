# Source this file on the HPC login shell to load Python and activate the venv.
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
module load numlib/cuDNN/9.5.0.50-CUDA-12.6.0

source "$HOME/aLoRA/.venv/bin/activate"

