#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
pip install protobuf==3.19.1
pip install ninja==1.11.1 sacrebleu==1.4.14 tqdm==4.50.2
pip install --editable ./G-Trans/.

if [ ! -d "./G-Trans/scripts_gtrans" ]; then
  cur_dir=$(pwd)
  ln -s $cur_dir/scripts_gtrans ./G-Trans/scripts_gtrans
fi