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
#   no_align -> $exp_nosep_path
#   token_align -> $exp_nosep_path
#   sent_align -> $exp_sep_path
data=$1
rawkd=$2
input=$3

exp_sep_path=exp_${rawkd}_sep
exp_nosep_path=exp_${rawkd}_nosep


# 1. Prepare dataset
# We assume the datasets have been prepared by running the scripts in ./scripts_at.

# 2. Train and evaluate NAT models
bash scripts_nat/setup-glat.sh

# no_align
bash scripts_nat/no_align/run-nat.sh $data train $exp_nosep_path $input
bash scripts_nat/no_align/run-nat.sh $data test $exp_nosep_path $input
bash scripts_nat/no_align/run-glat.sh $data train $exp_nosep_path $input
bash scripts_nat/no_align/run-glat.sh $data test $exp_nosep_path $input

# token_align
bash scripts_nat/token_align/run-nat-ctc.sh $data train $exp_nosep_path $input
bash scripts_nat/token_align/run-nat-ctc.sh $data test $exp_nosep_path $input
bash scripts_nat/token_align/run-glat-ctc.sh $data train $exp_nosep_path $input
bash scripts_nat/token_align/run-glat-ctc.sh $data test $exp_nosep_path $input

# sent_align
bash scripts_nat/sent_align/run-gtrans-glat.sh $data train $exp_sep_path $input
bash scripts_nat/sent_align/run-gtrans-glat.sh $data test $exp_sep_path $input
bash scripts_nat/sent_align/run-gtrans-glat-ctc.sh $data train $exp_sep_path $input
bash scripts_nat/sent_align/run-gtrans-glat-ctc.sh $data test $exp_sep_path $input
