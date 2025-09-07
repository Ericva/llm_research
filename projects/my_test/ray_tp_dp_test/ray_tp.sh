#/bin/bash
set -xe

CUDA_VISIBLE_DEVICES=0,1 python ray_tp.py