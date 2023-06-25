#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

umask 002
#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#pip config set global.index-url https://pypi.org/simple
pip config set global.trusted-host mirrors.aliyun.com
pip config set global.index-url  http://mirrors.aliyun.com/pypi/simple

# Experiments for discourse phenomena evaluation: data in $data_path and experiments in $exp_path
data_path=data-cadec
exp_path=exp_discourse_raw

# 1. Prepare dataset
bash scripts_disc/download-data.sh $data_path
bash scripts_disc/prepare-docdata.sh $data_path $exp_path
bash scripts_disc/prepare-scoredata.sh $data_path $exp_path

# 2. Train and evaluate G-Transformer (rnd.)
bash scripts_at/setup-gtransformer.sh
bash scripts_disc/run-gtrans.sh data $exp_path
bash scripts_disc/run-gtrans.sh train $exp_path
bash scripts_disc/run-gtrans.sh test $exp_path
bash scripts_disc/run-gtrans.sh score $exp_path

# filter invalid data which has target_len / source_len > 2.0
bash scripts_disc/filter-data.sh cadec $exp_path doc

# 3. Train and evaluate G-Trans+GLAT+CTC
bash scripts_nat/setup-glat.sh
bash scripts_disc/run-gtrans-glat-ctc.sh train $exp_path
bash scripts_disc/run-gtrans-glat-ctc.sh test $exp_path
bash scripts_disc/run-gtrans-glat-ctc.sh score $exp_path

# 4 Train and evaluate G-Trans+DAT
bash scripts_setup/setup-da-transformer.sh
bash scripts_disc/run-gtrans-dat.sh train $exp_path
bash scripts_disc/run-gtrans-dat.sh test $exp_path
bash scripts_disc/run-gtrans-dat.sh score $exp_path

