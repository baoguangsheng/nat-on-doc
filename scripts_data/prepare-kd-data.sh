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
src_path=$2  # use relative path
exp_path=$3  # use relative path
input=$4  # doc, sent

slang=en
tlang=de

# switch to submodule and convert the relative path to absolute
cur_dir=$(pwd)
src_path=$cur_dir/$src_path
exp_path=$cur_dir/$exp_path
cd ./G-Trans

if [ $input == "sent" ]; then
  mode=full
else
  mode=partial
fi

echo `date`, Preparing knowledge-distilled data ...
echo `date`, data: $data, mode: $mode, exp_path: $src_path, slang: $slang, tlang: $tlang
bin_path=$src_path/$data-$input.binarized.$slang-$tlang

run_path=$src_path/run-finetune
cp_path=$run_path/$data-$input.checkpoints.$slang-$tlang
res_path=$run_path/$data-$input.results.$slang-$tlang
doc_langs=$slang,$tlang
mkdir -p $res_path

for split in train; do
  echo `date`, Testing ${input}-level model on ${split} dataset...
  python -m fairseq_cli.generate $bin_path --path $cp_path/checkpoint_best.pt \
           --gen-subset $split --batch-size 16 --beam 5 --max-len-a 1.2 --max-len-b 10 \
           --task translation_doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
           --doc-mode $mode --tokenizer moses --remove-bpe --sacrebleu \
           --gen-output $res_path/$split > $run_path/gen-$split.$data-$input.$slang-$tlang.log 2>&1
done

echo `date`, data: $data, exp_path: $exp_path, slang: $slang, tlang: $tlang
src_seg_path=$src_path/$data-$input.segmented.$slang-$tlang
src_bin_path=$src_path/$data-$input.binarized.$slang-$tlang
src_res_path=$src_path/run-finetune/$data-$input.results.$slang-$tlang

seg_path=$exp_path/$data-$input.segmented.$slang-$tlang
bin_path=$exp_path/$data-$input.binarized.$slang-$tlang
mkdir -p $exp_path $seg_path $bin_path

echo `date`, Generate segmented files ...
# use generated target for train, but original target for test and valid
cp $src_seg_path/* $seg_path/. -rf
for D in train; do
  echo `date`, $D
  cp $src_seg_path/$D.$slang $seg_path/$D.$slang
  cp $src_seg_path/$D.$tlang $seg_path/$D.$tlang.ref0
  cp $src_res_path/$D.seg.gen $seg_path/$D.$tlang
  cp $src_res_path/$D.seg.ref $seg_path/$D.$tlang.ref1
  diff $seg_path/$D.$tlang.ref0 $seg_path/$D.$tlang.ref1 | wc -l
done

echo `date`, Generate binarized files ...
dict_path=$src_bin_path/dict.$slang.txt
python -m fairseq_cli.preprocess --task translation --source-lang $slang --target-lang $tlang \
       --trainpref $seg_path/train --validpref $seg_path/valid --testpref $seg_path/test --destdir $bin_path \
       --srcdict $dict_path --tgtdict $dict_path --workers 8

cd $cur_dir