# env_setting

新机器 / 容器的开发环境一键配置脚本，主要面向国内网络（使用清华 TUNA 镜像）。

## 文件清单

| 脚本 | 说明 |
| --- | --- |
| `set.conda.sh` | 配置 conda 清华镜像源 |
| `set.pip.sh` | 配置 pip 清华镜像源 |
| `set.ssh.sh` | 生成 / 配置 SSH key |
| `set.zsh.sh` | 安装并配置 zsh |

## 使用

按需单独执行，**先按自己环境改路径**（如 `set.conda.sh` 里的 miniconda 安装路径）：

```bash
bash set.conda.sh
bash set.pip.sh
bash set.ssh.sh
bash set.zsh.sh
```

> 脚本含写死的本地路径，仅作个人备忘，套用前请检查。
