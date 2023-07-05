#!/usr/bin/env bash
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 iwslt17 train exp_test"
    exit
fi

# run command
data=$1
mode=$2
exp_path=$3

slang=en
tlang=de

# switch to submodule and convert the relative path to absolute
cur_dir=$(pwd)
exp_path=$cur_dir/$exp_path
cd ./G-Trans

echo `date`, data: $data, mode: $mode, exp_path: $exp_path, slang: $slang, tlang: $tlang
bin_path=$exp_path/$data-doc.binarized.$slang-$tlang

run_path=$exp_path/run-gtrans-default
mkdir -p $run_path
echo `date`, run path: $run_path

cp_path=$run_path/$data-doc.checkpoints.$slang-$tlang
res_path=$run_path/$data-doc.results.$slang-$tlang

if [ $mode == "train" ]; then
  echo `date`, Training model...
  doc_langs=$slang,$tlang
  python train.py $bin_path --save-dir $cp_path --tensorboard-logdir $cp_path --seed 555 --fp16 --num-workers 4 \
         --task translation_doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
         --arch gtransformer_base --doc-mode partial --share-all-embeddings \
         --optimizer adam --adam-betas "(0.9, 0.98)" --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --no-epoch-checkpoints \
         --max-tokens 4096 --update-freq 1 --validate-interval 1 --patience 10 \
         --encoder-ctxlayers 2 --decoder-ctxlayers 2 --cross-ctxlayers 2 \
         --doc-noise-mask 0.3 --doc-noise-epochs 30 > $run_path/train.$data-doc.$slang-$tlang.log 2>&1

elif [ $mode == "test" ]; then
  mkdir -p $res_path
  doc_langs=$slang,$tlang
  echo `date`, Testing model on test dataset...
  python -m fairseq_cli.generate $bin_path --path $cp_path/checkpoint_best.pt \
         --gen-subset test --batch-size 16 --beam 5 --max-len-a 1.2 --max-len-b 10 \
         --task translation_doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
         --doc-mode partial --tokenizer moses --remove-bpe --sacrebleu \
         --gen-output $res_path/test > $run_path/test.$data-doc.$slang-$tlang.log 2>&1

else
  echo Unknown mode ${mode}.
fi

cd $cur_dir