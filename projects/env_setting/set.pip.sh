#/bin/bash

python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

export HF_ENDPOINT=https://hf-mirror.com
export https_proxy=xxx.xxx.xxx.xxx:xxxx