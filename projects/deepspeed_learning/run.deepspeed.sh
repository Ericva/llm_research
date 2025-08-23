#/bin/bash
set -x
torchrun --nproc_per_node=2 deepspeed.1.py --deepspeed --deepspeed_config ds.config