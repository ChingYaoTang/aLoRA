# Source this file on the HPC login shell to load Python and activate the venv.
module load ai/PyTorch/2.3.0-foss-2023b
# module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
# module load numlib/cuDNN/9.5.0.50-CUDA-12.6.0
# python3 -m venv $HOME/aLoRA/.venv --system-site-packages

source "$HOME/aLoRA/.venv/bin/activate"
# python -m pip install -U pip
# python -m pip install peft accelerate trl datasets
# python -m pip install bitsandbytes

