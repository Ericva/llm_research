#!/bin/bash
set -x

apt-get update && \
apt-get install -y openssh-server && \
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
echo "Port xxxx" >> /etc/ssh/sshd_config && \
/etc/init.d/ssh  start && \
mkdir -p /root/.ssh