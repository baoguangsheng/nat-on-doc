#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 iwslt17 exp_final"
    exit
fi

# setup the environment
set -e  # exit if error
umask 002  # avoid root privilege in docker

#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#pip config set global.index-url https://pypi.org/simple
pip config set global.trusted-host mirrors.aliyun.com
pip config set global.index-url  http://mirrors.aliyun.com/pypi/simple

data=$1
exp_root=$2
input=doc

# 1. Train and evaluate Transformer
# setup
bash -e scripts_nat/setup-glat.sh

# raw data
exp_path=$exp_root/exp_raw_nosep
bash -e scripts_at/run-transformer.sh $data train $exp_path $input
bash -e scripts_at/run-transformer.sh $data test $exp_path $input

# KD data
exp_path=$exp_root/exp_kd_nosep
bash -e scripts_at/run-transformer.sh $data train $exp_path $input
bash -e scripts_at/run-transformer.sh $data test $exp_path $input

# 2. Train and evaluate G-Transformer
# setup
bash -e scripts_data/setup-gtransformer.sh

# raw data
# skip this step since we have already trained G-Transformer on raw data during data preparation.

# KD data
exp_path=$exp_root/exp_kd
bash -e scripts_at/run-gtrans-randinit.sh $data train $exp_path $input
bash -e scripts_at/run-gtrans-randinit.sh $data test $exp_path $input
