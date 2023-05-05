#!/usr/bin/env bash
# bash prepare-docdata.sh data-cadec exp_disc_raw
data_path=$1
exp_path=$2

code=bpe
slang=en
tlang=ru
data=cadec

echo `date`, exp_path: $exp_path, input: doc, code: $code, slang: $slang, tlang: $tlang
mkdir -p $exp_path
tok_path=$exp_path/$data.tokenized.$slang-$tlang
seg_path=$exp_path/$data-doc.segmented.$slang-$tlang

echo `date`, Prepraring data...
# tokenize and sub-word
bash scripts_disc/prepare-docbpe.sh $data_path/sent $data_path/doc $tok_path

# data builder
python -m scripts_disc.data_builder --datadir $tok_path --destdir $seg_path/ --source-lang $slang --target-lang $tlang --max-tokens 512 --max-sents 1000
