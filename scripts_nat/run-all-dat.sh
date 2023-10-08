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

# Setup env
bash scripts_nat/setup-da-transformer.sh

# token-level alignment
exp_path=$exp_root/exp_${data_type}_nosep/exp_filtered
bash -e scripts_nat/run-da-transformer.sh $data train $exp_path $input
bash -e scripts_nat/run-da-transformer.sh $data test $exp_path $input

# sent-level alignment
exp_path=$exp_root/exp_${data_type}/exp_filtered
bash -e scripts_nat/run-gtrans-dat.sh $data train $exp_path $input
bash -e scripts_nat/run-gtrans-dat.sh $data test $exp_path $input
