#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 iwslt17 exp_root"
    exit
fi

data=$1
exp_root=$2
input=doc

# setup the environment
set -e  # exit if error
umask 002  # avoid root privilege in docker

# 1. Prepare environment for the model
bash -e scripts_data/setup-gtransformer.sh

# 2. Prepare raw datasets and train AT model for knowledge distillation
exp_path=$exp_root/exp_raw
bash -e scripts_data/run-gtrans-finetune.sh $data data $exp_path
bash -e scripts_data/run-gtrans-finetune.sh $data train $exp_path
bash -e scripts_data/run-gtrans-finetune.sh $data test $exp_path

# 3. Knowledge distillation from the trained models
from_exp_path=$exp_root/exp_raw
exp_path=$exp_root/exp_kd
bash -e scripts_data/prepare-kd-data.sh $data $from_exp_path $exp_path $input

# 4. Remove sentence separator for none G-Trans models
from_exp_path=$exp_root/exp_raw
exp_path=$exp_root/exp_raw_nosep
bash -e scripts_data/remove-sentseparator.sh $data $from_exp_path $exp_path $input

from_exp_path=$exp_root/exp_kd
exp_path=$exp_root/exp_kd_noseq
bash -e scripts_data/remove-sentseparator.sh $data $from_exp_path $exp_path $input
