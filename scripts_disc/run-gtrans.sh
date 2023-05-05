#!/usr/bin/env bash
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 train exp_test"
    exit
fi

# run command
mode=$1
exp_path=$2
doc=partial
seed=666

data=cadec
slang=en
tlang=ru
doc_langs=$slang,$tlang
cadec_path=CADec

# switch to submodule and convert the relative path to absolute
cur_dir=$(pwd)
exp_path=$cur_dir/$exp_path
cadec_path=$cur_dir/$cadec_path
cd ./G-Trans

# run on the submodule
run_path=$exp_path/run-gtrans-default
mkdir -p $run_path
echo `date`, run path: $run_path
echo `date`, data: $data, mode: $mode, exp_path: $exp_path, doc: $doc, slang: $slang, tlang: $tlang

if [ $mode == "data" ]; then
  echo `date`, Binarizing $data data...
  seg_path=$exp_path/$data-doc.segmented.$slang-$tlang
  bin_path=$exp_path/$data-doc.binarized.$slang-$tlang
  python -m fairseq_cli.preprocess --task translation_doc --source-lang $slang --target-lang $tlang \
     --trainpref $seg_path/train --validpref $seg_path/valid --testpref $seg_path/test --destdir $bin_path \
     --joined-dictionary --workers 8

  echo `date`, Binarizing score data...
  src_dict_path=$bin_path/dict.en.txt
  data=score
  seg_path=$exp_path/$data-doc.segmented.$slang-$tlang
  bin_path=$exp_path/$data-doc.binarized.$slang-$tlang
  for D in deixis ellinfl ellvp lexcohe; do
    python -m fairseq_cli.preprocess --task translation_doc --source-lang $slang --target-lang $tlang \
           --testpref $seg_path/$D --destdir $bin_path --srcdict $src_dict_path --tgtdict $src_dict_path --workers 8
    # rename the default test
    mv $bin_path/test.$slang-$tlang.$slang.bin $bin_path/$D.$slang-$tlang.$slang.bin
    mv $bin_path/test.$slang-$tlang.$slang.idx $bin_path/$D.$slang-$tlang.$slang.idx
    mv $bin_path/test.$slang-$tlang.$tlang.bin $bin_path/$D.$slang-$tlang.$tlang.bin
    mv $bin_path/test.$slang-$tlang.$tlang.idx $bin_path/$D.$slang-$tlang.$tlang.idx
  done

elif [ $mode == "train" ]; then
  echo `date`, Training model...
  bin_path=$exp_path/$data-doc.binarized.$slang-$tlang
  cp_path=$run_path/$data-doc.checkpoints.$slang-$tlang
  python train.py $bin_path --save-dir $cp_path --tensorboard-logdir $cp_path --seed $seed --fp16 --num-workers 4 \
         --task translation_doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
         --arch gtransformer_base --doc-mode $doc --share-all-embeddings \
         --optimizer adam --adam-betas "(0.9, 0.98)" --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
         --no-epoch-checkpoints --save-interval-updates 1000 --keep-interval-updates 1 \
         --max-tokens 4096 --update-freq 2 --validate-interval 1 --patience 10 \
         --encoder-ctxlayers 2 --decoder-ctxlayers 2 --cross-ctxlayers 2 \
         --doc-noise-mask 0.3 --doc-noise-epochs 30 > $run_path/train.$data.$slang-$tlang.log 2>&1

elif [ $mode == "test" ]; then
  echo `date`, Testing model on test dataset...
  bin_path=$exp_path/$data-doc.binarized.$slang-$tlang
  cp_path=$run_path/$data-doc.checkpoints.$slang-$tlang
  res_path=$run_path/$data-doc.results.$slang-$tlang
  mkdir -p $res_path
  python -m fairseq_cli.generate $bin_path --path $cp_path/checkpoint_best.pt \
         --gen-subset test --batch-size 16 --beam 5 --max-len-a 1.2 --max-len-b 10 \
         --task translation_doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
          --doc-mode $doc --tokenizer moses --remove-bpe --sacrebleu \
         --gen-output $res_path/test.$data.$slang-$tlang > $run_path/test.$data.$slang-$tlang.log 2>&1

elif [ $mode == "score" ]; then
  echo `date`, Scoring model on deixis,ellinfl,ellvp,lexcohe datasets...
  cp_path=$run_path/$data-doc.checkpoints.$slang-$tlang
  data=score
  bin_path=$exp_path/$data-doc.binarized.$slang-$tlang
  res_path=$run_path/$data-doc.results.$slang-$tlang
  mkdir -p $res_path
  python -m fairseq_cli.validate $bin_path --path $cp_path/checkpoint_best.pt \
         --valid-subset deixis,ellinfl,ellvp,lexcohe --batch-size 1 \
         --task translation_doc --doc-mode $doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
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