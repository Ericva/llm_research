#/bin/bash
set -x
torchrun --nproc_per_node=2 accelerate.1.py

# or
# accelerate launch accelerate.1.py

# or
# python -m accelerate.commands.launch accelerate.1.py