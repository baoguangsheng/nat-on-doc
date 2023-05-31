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
bash scripts_setup/setup-gtransformer.sh

# 2. Prepare raw datasets and train AT models
exp_path=exp_sentalign_raw
#bash scripts_at/run-gtransformer.sh $data data $exp_path
bash scripts_at/run-gtransformer.sh $data train $exp_path
bash scripts_at/run-gtransformer.sh $data test $exp_path

# 3. Knowledge distillation from the trained models
from_exp_path=exp_sentalign_raw
exp_path=exp_sentalign_kd
bash scripts_at/prepare-kd-data.sh $data $from_exp_path $exp_path $input

# 4. Remove sentence separator for none G-Trans models
from_exp_path=exp_sentalign_raw
exp_path=exp_otheralign_raw
bash scripts_at/remove-sentalign.sh $data $from_exp_path $exp_path $input

from_exp_path=exp_sentalign_kd
exp_path=exp_otheralign_kd
bash scripts_at/remove-sentalign.sh $data $from_exp_path $exp_path $input

