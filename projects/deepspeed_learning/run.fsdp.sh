#/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 fsdp.1.py