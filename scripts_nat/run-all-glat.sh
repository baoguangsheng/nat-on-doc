#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 iwslt17 exp_root raw/kd"
    exit
fi

# setup the environment
set -e  # exit if error
umask 002  # avoid root privilege in docker

data=$1
exp_root=$2
data_type=$3  # data type
input=doc

# Setup
bash scripts_nat/setup-glat.sh

# implicit alignment
exp_path=$exp_root/exp_${data_type}_nosep
bash scripts_nat/run-nat.sh $data train $exp_path $input
bash scripts_nat/run-nat.sh $data test $exp_path $input
bash scripts_nat/run-glat.sh $data train $exp_path $input
bash scripts_nat/run-glat.sh $data test $exp_path $input

# token-level alignment
exp_path=$exp_root/exp_${data_type}_nosep/exp_filtered
bash scripts_nat/run-nat-ctc.sh $data train $exp_path $input
bash scripts_nat/run-nat-ctc.sh $data test $exp_path $input
bash scripts_nat/run-glat-ctc.sh $data train $exp_path $input
bash scripts_nat/run-glat-ctc.sh $data test $exp_path $input

# sent-level alignment
exp_path=$exp_root/exp_${data_type}
bash scripts_nat/run-gtrans-glat.sh $data train $exp_path $input
bash scripts_nat/run-gtrans-glat.sh $data test $exp_path $input

exp_path=$exp_root/exp_${data_type}/exp_filtered
bash scripts_nat/run-gtrans-glat-ctc.sh $data train $exp_path $input
bash scripts_nat/run-gtrans-glat-ctc.sh $data test $exp_path $input
