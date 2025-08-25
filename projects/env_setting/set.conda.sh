#!/bin/bash
set -x

source /root/xxx/xxx/miniconda3/etc/profile.d/conda.sh

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud//pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes

conda config --show channels


if [[  -f ~/.zshrc  ]];then
cat <<- 'EOF' >> ~/.zshrc

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/xxx/xxx/xxx/install/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/xxx/xxx/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/xxx/xxx/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/xxx/xxx/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
EOF
fi