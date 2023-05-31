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
NUMEXPR_MAX_THREADS=8

# switch to submodule and convert the relative path to absolute
cur_dir=$(pwd)
exp_path=$cur_dir/$exp_path
cd ./DA-Trans

echo `date`, data: $data, mode: $mode, exp_path: $exp_path, slang: $slang, tlang: $tlang
bin_path=$exp_path/$data-$input.binarized.$slang-$tlang

run_path=$exp_path/run-da-transformer-default
mkdir -p $run_path
echo `date`, run path: $run_path

cp_path=$run_path/$data-$input.checkpoints.$slang-$tlang
res_path=$run_path/$data-$input.results.$slang-$tlang

if [ $input == 'sent' ]; then
  upsample_scale=4.0
else
  upsample_scale=2.0
fi

if [ $mode == "train" ]; then
  echo `date`, Training model...
  python train.py $bin_path  --user-dir fs_plugins --task translation_lev_modified  --noise full_mask \
      --arch glat_decomposed_link_base --decoder-learned-pos --encoder-learned-pos --share-all-embeddings --activation-fn gelu \
      --apply-bert-init --links-feature feature:position --decode-strategy lookahead \
      --max-source-positions 512 --max-target-positions 1024 --src-upsample-scale $upsample_scale \
      --criterion nat_dag_loss --length-loss-factor 0 --max-transition-length 99999 \
      --glat-p 0.5:0.1@200k --glance-strategy number-random --optimizer adam --adam-betas '(0.9,0.999)' --fp16 \
      --label-smoothing 0.0 --weight-decay 0.01 --dropout 0.1 --lr-scheduler inverse_sqrt  --warmup-updates 10000   \
      --clip-norm 0.1 --lr 0.0005 --warmup-init-lr '1e-07' --stop-min-lr '1e-09' --ddp-backend c10d \
      --grouped-shuffling --seed 0 --valid-subset valid --skip-invalid-size-inputs-valid-test \
      --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
      --eval-bleu-detok space --eval-bleu-remove-bpe --eval-bleu-print-samples --fixed-validation-seed 7 \
      --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  --save-dir $cp_path \
      --no-epoch-checkpoints --no-last-checkpoints \
      --max-tokens 4096 --update-freq 1 --validate-interval 1 --patience 30 \
      > $run_path/train.$data-$input.$slang-$tlang.log 2>&1

elif [ $mode == "test" ]; then
  echo `date`, Testing model on test dataset...
  # look ahead
  fairseq-generate $bin_path --source-lang $slang --target-lang $tlang \
    --gen-subset test --user-dir fs_plugins --task translation_lev_modified \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
    --max-tokens 4096 --seed 0 \
    --model-overrides "{\"decode_strategy\":\"lookahead\",\"decode_beta\":1}" \
    --path $cp_path/checkpoint_best.pt \
    --remove-bpe --tokenizer moses --sacrebleu --scoring sacrebleu \
    > $run_path/test.$data-$input.$slang-$tlang.log 2>&1

  # beam search
#  fairseq-generate $bin_path \
#      --gen-subset test --user-dir fs_plugins --task translation_lev_modified \
#      --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
#      --remove-bpe --batch-size 32 --seed 0 \
#      --model-overrides "{\"decode_strategy\": \"beamsearch\", \"decode_beta\": 1, \
#          \"decode_alpha\": 1.1, \"decode_gamma\": 0, \
#          \"decode_lm_path\": None, \
#          \"decode_beamsize\": 200, \"decode_top_cand_n\": 5, \"decode_top_p\": 0.9, \
#          \"decode_max_beam_per_length\": 10, \"decode_max_batchsize\": 32, \"decode_dedup\": True}" \
#      --path $cp_path/checkpoint_best.pt > $run_path/test.$data-$input.$slang-$tlang.log 2>&1

else
  echo Unknown mode ${mode}.
fi

cd $cur_dir
