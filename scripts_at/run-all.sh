#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 iwslt17"
    exit
fi

umask 002
pip config set global.trusted-host mirrors.aliyun.com
pip config set global.index-url  http://mirrors.aliyun.com/pypi/simple

data=$1
input=doc

# 1. Prepare environment for the model
bash scripts_at/setup-gtrans.sh

# 2. Prepare raw datasets and train AT models
exp_path=exp_raw_sep  # raw data with separator between sentences
#bash scripts_at/run-gtrans.sh $data data $exp_path
bash scripts_at/run-gtrans.sh $data train $exp_path
bash scripts_at/run-gtrans.sh $data test $exp_path

# 3. Knowledge distillation from the trained models
from_exp_path=exp_raw_sep
exp_path=exp_kd_sep
bash scripts_at/prepare-data-kd.sh $data $from_exp_path $exp_path $input

# 4. Remove sentence separator for none G-Trans models
from_exp_path=exp_raw_sep
exp_path=exp_raw_nosep
bash scripts_at/prepare-data-nosep.sh $data $from_exp_path $exp_path $input

from_exp_path=exp_kd_sep
exp_path=exp_kd_nosep
bash scripts_at/prepare-data-nosep.sh $data $from_exp_path $exp_path $input

