#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 iwslt17 raw/kd doc"
    exit
fi

umask 002
#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#pip config set global.index-url https://pypi.org/simple
pip config set global.trusted-host mirrors.aliyun.com
pip config set global.index-url  http://mirrors.aliyun.com/pypi/simple

# Experiments: sent-align uses data with sentence separator
#   no_align -> $exp_otheralign_path
#   token_align -> $exp_otheralign_path
#   sent_align -> $exp_sentalign_path
data=$1
raw_kd=$2
input=$3

exp_sentalign_path=exp_sentalign_${raw_kd}
exp_otheralign_path=exp_otheralign_${raw_kd}


# 1. Prepare dataset
# We assume the datasets have been prepared by running the scripts in ./scripts_at.

# 2. Train and evaluate NAT models
bash scripts_nat/setup-da-transformer.sh

# token_align
bash scripts_nat/token_align/run-da-transformer.sh $data train $exp_otheralign_path $input
bash scripts_nat/token_align/run-da-transformer.sh $data test $exp_otheralign_path $input

# sent_align
bash scripts_nat/sent_align/run-gtrans-dat.sh $data train $exp_sentalign_path $input
bash scripts_nat/sent_align/run-gtrans-dat.sh $data test $exp_sentalign_path $input
