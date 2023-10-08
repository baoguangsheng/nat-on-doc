#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 iwslt17 from_exp_test exp_test doc"
    exit
fi

# run command
data=$1
src_path=$2
exp_path=$3
input=$4  # doc, sent

slang=en
tlang=de

# switch to submodule and convert the relative path to absolute
cur_dir=$(pwd)
src_path=$cur_dir/$src_path
exp_path=$cur_dir/$exp_path
cd ./G-Trans

echo `date`, data: $data, exp_path: $exp_path, slang: $slang, tlang: $tlang
seg_path=$exp_path/$data-$input.segmented.$slang-$tlang
bin_path=$exp_path/$data-$input.binarized.$slang-$tlang

mkdir -p $exp_path
rm $exp_path/* -rf
cp $src_path/$data-$input.* $exp_path/. -rf

echo `date`, Filter invalid data which target_len / source_len > 2.0 ...
python ../scripts_at/filter_data.py --data-path $seg_path --slang $slang --tlang $tlang

echo `date`, Generate binarized files ...
dict_path=$bin_path/dict.$slang.txt

# remove old files
rm $bin_path/train.* -f
rm $bin_path/valid.* -f
rm $bin_path/test.* -f

python -m fairseq_cli.preprocess --task translation --source-lang $slang --target-lang $tlang \
       --trainpref $seg_path/train --validpref $seg_path/valid --testpref $seg_path/test --destdir $bin_path \
       --srcdict $dict_path --tgtdict $dict_path --workers 8

cd $cur_dir
