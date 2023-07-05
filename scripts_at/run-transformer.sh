#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 iwslt17 train exp_test doc"
    echo "    bash $0 iwslt17 test exp_test doc"
    exit
fi

# run command
data=$1
mode=$2  # train, test
exp_path=$3
input=$4  # doc, sent

slang=en
tlang=de

# switch to submodule and convert the relative path to absolute
cur_dir=$(pwd)
exp_path=$cur_dir/$exp_path
cd ./GLAT

# run on the submodule
run_path=$exp_path/run-transformer-default
mkdir -p $run_path
echo `date`, run path: $run_path
echo `date`, data: $data, mode: $mode, exp_path: $exp_path, slang: $slang, tlang: $tlang

bin_path=$exp_path/$data-$input.binarized.$slang-$tlang
cp_path=$run_path/$data-$input.checkpoints.$slang-$tlang

if [ $mode == "train" ]; then
  echo `date`, Training model...
  python train.py $bin_path --save-dir $cp_path --tensorboard-logdir $cp_path --seed 555 --fp16 --num-workers 4 \
      --task translation --source-lang $slang --target-lang $tlang \
      --arch transformer --dropout 0.3 --share-all-embeddings \
      --optimizer adam --adam-betas "(0.9, 0.98)" --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --no-epoch-checkpoints \
      --max-tokens 4096 --update-freq 1 --validate-interval 1 --patience 10 \
      > $run_path/train.$data-$input.$slang-$tlang.log 2>&1

elif [ $mode == "test" ]; then
  echo `date`, Testing model on test dataset...
  fairseq-generate $bin_path --path $cp_path/checkpoint_best.pt  \
      --gen-subset test --batch-size 1 --beam 5 --max-len-a 1.2 --max-len-b 10 \
      --task translation --source-lang $slang --target-lang $tlang  \
      --remove-bpe --tokenizer moses --sacrebleu --scoring sacrebleu \
      > $run_path/test.$data-$input.$slang-$tlang.log 2>&1

else
  echo Unknown mode ${mode}.
fi

cd $cur_dir
