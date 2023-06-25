#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 train exp_test"
    echo "    bash $0 test exp_test"
    exit
fi

# run command
mode=$1  # train, test
exp_path=$2

data=cadec
slang=en
tlang=ru
cadec_path=CADec
plugin_path=plugins_gtrans
upsample_scale=2

# switch to submodule and convert the relative path to absolute
cur_dir=$(pwd)
exp_path=$cur_dir/$exp_path
cadec_path=$cur_dir/$cadec_path
plugin_path=$cur_dir/$plugin_path
cd ./DA-Trans

# run on the submodule
run_path=$exp_path/run-gtrans-dat-default
mkdir -p $run_path
echo `date`, run path: $run_path
echo `date`, data: $data, mode: $mode, exp_path: $exp_path, slang: $slang, tlang: $tlang

if [ $mode == "train" ]; then
  echo `date`, Training model...
  bin_path=$exp_path/$data-doc.binarized.$slang-$tlang
  cp_path=$run_path/$data-doc.checkpoints.$slang-$tlang
  CUDA_LAUNCH_BLOCKING=1 python train.py $bin_path  --user-dir $plugin_path --task translation_lev_modified_doc  --noise full_mask \
      --arch gtrans_glat_decomposed_link_base --decoder-learned-pos --encoder-learned-pos --share-all-embeddings --activation-fn gelu \
      --apply-bert-init --links-feature feature:position --decode-strategy lookahead \
      --max-source-positions 512 --max-target-positions $(($upsample_scale * 512)) --src-upsample-scale $upsample_scale \
      --criterion nat_dag_loss --length-loss-factor 0 --max-transition-length 99999 \
      --glat-p 0.5:0.1@200k --glance-strategy number-random --optimizer adam --adam-betas '(0.9,0.999)' --fp16 \
      --label-smoothing 0.0 --weight-decay 0.01 --dropout 0.1 --lr-scheduler inverse_sqrt  --warmup-updates 10000   \
      --clip-norm 0.1 --lr 0.0005 --warmup-init-lr '1e-07' --stop-min-lr '1e-09' --ddp-backend c10d \
      --grouped-shuffling --seed 0 --valid-subset valid --skip-invalid-size-inputs-valid-test \
      --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
      --eval-bleu-detok space --eval-bleu-remove-bpe --eval-bleu-print-samples --fixed-validation-seed 7 \
      --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  --save-dir $cp_path \
      --no-epoch-checkpoints --no-last-checkpoints \
      --save-interval-updates 1000 --keep-interval-updates 1 \
      --max-tokens 4096 --update-freq 2 --validate-interval 1 --patience 30 \
      --doc-mode partial --encoder-ctxlayers 2 --decoder-ctxlayers 2 --cross-ctxlayers 2 \
      --source-lang $slang --target-lang $tlang \
      > $run_path/train.$data-doc.$slang-$tlang.log 2>&1

elif [ $mode == "test" ]; then
  echo `date`, Testing model on test dataset...
  bin_path=$exp_path/$data-doc.binarized.$slang-$tlang
  cp_path=$run_path/$data-doc.checkpoints.$slang-$tlang
  # look ahead
  PYTHONUNBUFFERED='True' python -m fairseq_cli.generate2 $bin_path --source-lang $slang --target-lang $tlang \
    --gen-subset test --user-dir $plugin_path --task translation_lev_modified_doc \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
    --max-tokens 4096 --seed 0 \
    --model-overrides "{\"decode_strategy\":\"lookahead\",\"decode_beta\":1}" \
    --path $cp_path/checkpoint_best.pt \
    --remove-bpe --tokenizer moses --sacrebleu --scoring sacrebleu \
    --doc-mode partial \
    > $run_path/test.$data-doc.$slang-$tlang.log 2>&1

elif [ $mode == "score" ]; then
  echo `date`, Scoring model on deixis,ellinfl,ellvp,lexcohe datasets...
  cp_path=$run_path/$data-doc.checkpoints.$slang-$tlang
  data=score
  bin_path=$exp_path/$data-doc.binarized.$slang-$tlang
  res_path=$run_path/$data-doc.results.$slang-$tlang
  mkdir -p $res_path

  python -m fairseq_cli.validate2 $bin_path --path $cp_path/checkpoint_best.pt --user-dir $plugin_path \
         --task translation_lev_modified_doc --max-sentences 20 --source-lang $slang --target-lang $tlang \
         --doc-mode partial --valid-subset deixis,ellinfl,ellvp,lexcohe --batch-size 1 \
         --gen-output $res_path/valid.$data.$slang-$tlang > $run_path/valid.$data.$slang-$tlang.log 2>&1

  echo `date`, ---- deixis_test ---- > $run_path/test.$data.$slang-$tlang.log 2>&1
  cut -f4 $res_path/valid.$data.$slang-$tlang.deixis.score > $res_path/deixis.score
  python $cadec_path/scripts/evaluate_consistency.py --repo-dir $cadec_path --test deixis_test \
         --scores $res_path/deixis.score >> $run_path/test.$data.$slang-$tlang.log 2>&1

  echo `date`, ---- ellipsis_infl ---- >> $run_path/test.$data.$slang-$tlang.log 2>&1
  cut -f4 $res_path/valid.$data.$slang-$tlang.ellinfl.score > $res_path/ellinfl.score
  python $cadec_path/scripts/evaluate_consistency.py --repo-dir $cadec_path --test ellipsis_infl \
         --scores $res_path/ellinfl.score >> $run_path/test.$data.$slang-$tlang.log 2>&1

  echo `date`, ---- ellipsis_vp ---- >> $run_path/test.$data.$slang-$tlang.log 2>&1
  cut -f4 $res_path/valid.$data.$slang-$tlang.ellvp.score > $res_path/ellvp.score
  python $cadec_path/scripts/evaluate_consistency.py --repo-dir $cadec_path --test ellipsis_vp \
         --scores $res_path/ellvp.score >> $run_path/test.$data.$slang-$tlang.log 2>&1

  echo `date`, ---- lex_cohesion_test ---- >> $run_path/test.$data.$slang-$tlang.log 2>&1
  cut -f4 $res_path/valid.$data.$slang-$tlang.lexcohe.score > $res_path/lexcohe.score
  python $cadec_path/scripts/evaluate_consistency.py --repo-dir $cadec_path --test lex_cohesion_test \
         --scores $res_path/lexcohe.score >> $run_path/test.$data.$slang-$tlang.log 2>&1

else
  echo Unknown mode ${mode}.
fi

cd $cur_dir
