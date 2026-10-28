# This is intented to be run with sudo

nvidia-smi --lock-gpu-clocks=2550,3000
nvidia-smi --lock-memory-clocks=10251

# First setup python env in sudo
source pyenv/bin/activate
# Then run the script
python profiling.py > prof_results.txt

nvidia-smi --reset-memory-clocks
nvidia-smi --reset-gpu-clocks
