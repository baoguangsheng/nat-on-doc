#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# DA-Transformer requires pytorch1.10 and above.
cp ./plugins_gtrans/generate.py ./DA-Trans/fairseq_cli/generate2.py -f
cp ./plugins_gtrans/validate.py ./DA-Trans/fairseq_cli/validate2.py -f

pip install protobuf==3.19.1
pip install ninja==1.11.1 sacrebleu==1.4.14 tensorboard==2.6.0
pip install --editable ./DA-Trans/.

cur_dir=$(pwd)
cd ./DA-Trans/DAG-Search-main
bash install.sh
cd $cur_dir