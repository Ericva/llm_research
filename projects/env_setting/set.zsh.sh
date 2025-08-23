#!/bin/bash
# 设置脚本错误时终止执行
set -e

echo "Starting environment setup..."

# 更新 apt-get 包
echo "Running apt-get update..."
apt-get update
echo "apt-get update completed."

# 安装 zsh
echo "Installing zsh..."
apt-get install -y zsh
echo "zsh installation completed."

# 安装 vim
echo "Installing vim..."
apt-get install -y vim
echo "vim installation completed."

# 解压并安装 vim 配置
echo "Extracting vim configuration from vim.tar.gz..."
tar -zxvf vim.tar.gz ./.vim
echo "vim configuration extracted."

echo "Moving vim configuration to the home directory..."
mv ./.vim ~/.vim/
echo "vim configuration moved to ~/.vim."

echo "Copying vimrc to ~/.vimrc..."
cp ./vimrc ~/.vimrc
echo "vimrc copied."

# 安装 oh-my-zsh
echo "Cloning oh-my-zsh repository..."
rm -rf  ~/.oh-my-zsh
git clone https://github.com/ohmyzsh/ohmyzsh.git ~/.oh-my-zsh
echo "oh-my-zsh repository cloned."

# Backup existing zshrc file
echo "Backing up existing .zshrc file to .zshrc.orig..."
[ -f ~/.zshrc ] && cp ~/.zshrc ~/.zshrc.orig
echo "Existing .zshrc file backed up."

# Copy the template zshrc to the home directory
echo "Copying oh-my-zsh template zshrc..."
cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
echo "Template zshrc copied."

# 更改默认 shell 为 zsh
echo "Changing default shell to zsh..."
chsh -s $(which zsh)
echo "Default shell changed to zsh."

# 打印完成消息
echo "Environment setup completed. Please restart your terminal or run 'source ~/.zshrc' to apply changes."