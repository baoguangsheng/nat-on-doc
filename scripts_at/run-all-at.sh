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

# 1. Train and evaluate Transformer
bash -e scripts_nat/setup-glat.sh

exp_path=$exp_root/exp_${data_type}_nosep
bash -e scripts_at/run-transformer.sh $data train $exp_path $input
bash -e scripts_at/run-transformer.sh $data test $exp_path $input

# 2. Train and evaluate G-Transformer
bash -e scripts_data/setup-gtransformer.sh

if [[ $data_type == 'raw' ]]; then
  echo `date`, Skip it since we have already trained G-Transformer on raw data during data preparation.
else
  exp_path=$exp_root/exp_${data_type}
  bash -e scripts_at/run-gtrans-randinit.sh $data train $exp_path $input
  bash -e scripts_at/run-gtrans-randinit.sh $data test $exp_path $input
fi
