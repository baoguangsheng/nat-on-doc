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
run_path=$exp_path/run-glat-default
mkdir -p $run_path
echo `date`, run path: $run_path
echo `date`, data: $data, mode: $mode, exp_path: $exp_path, slang: $slang, tlang: $tlang

bin_path=$exp_path/$data-$input.binarized.$slang-$tlang
cp_path=$run_path/$data-$input.checkpoints.$slang-$tlang

plugin_path=glat_plugins
if [ $mode == "train" ]; then
  echo `date`, Training model...
  python train.py $bin_path --arch glat --noise full_mask --share-all-embeddings --seed 0 --fp16 \
      --criterion glat_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
      --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
      --adam-eps 1e-6 --task translation_lev_modified --weight-decay 0.01 --dropout 0.1 \
      --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 \
      --max-source-positions 1024 --max-target-positions 1024 --clip-norm 5   \
      --save-dir $cp_path --src-embedding-copy --length-loss-factor 0.05 --log-interval 1000 \
      --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
      --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
      --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
      --apply-bert-init --activation-fn gelu --user-dir $plugin_path \
      --no-epoch-checkpoints --no-last-checkpoints \
      --max-tokens 4096 --update-freq 1 --validate-interval 1 --patience 30 \
      > $run_path/train.$data-$input.$slang-$tlang.log 2>&1

elif [ $mode == "test" ]; then
  echo `date`, Testing model on test dataset...
  fairseq-generate $bin_path --path $cp_path/checkpoint_best.pt --user-dir $plugin_path \
      --task translation_lev_modified --max-sentences 20 --source-lang $slang --target-lang $tlang \
      --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test \
      --remove-bpe --tokenizer moses --sacrebleu  --scoring sacrebleu \
      > $run_path/test.$data-$input.$slang-$tlang.log 2>&1

else
  echo Unknown mode ${mode}.
fi

cd $cur_dir
