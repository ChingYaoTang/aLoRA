# Source this file on the HPC login shell to load Python and activate the venv.
module load ai/PyTorch/2.3.0-foss-2023b
# python3 -m venv $HOME/aLoRA/.venv --system-site-packages

source "$HOME/aLoRA/.venv/bin/activate"
# python -m pip install -U pip
# python -m pip install bitsandbytes

