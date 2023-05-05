#!/usr/bin/env bash
# bash prepare-scoredata.sh data-cadec exp_disc_raw
data_path=$1
exp_path=$2

slang=en
tlang=ru

echo `date`, exp_path: $exp_path, input: doc, code: $code, slang: $slang, tlang: $tlang
tok_path=$exp_path/score.tokenized.$slang-$tlang
seg_path=$exp_path/score-doc.segmented.$slang-$tlang

echo `date`, Prepraring data...
# tokenize and sub-word
src_tok_path=$exp_path/cadec.tokenized.$slang-$tlang
bash scripts_disc/prepare-scorebpe.sh $data_path/score $tok_path $src_tok_path

# data builder
python -m scripts_disc.data_builder --corpuses deixis,ellinfl,ellvp,lexcohe --datadir $tok_path --destdir $seg_path/ --source-lang $slang --target-lang $tlang --max-tokens 512 --max-sents 1000

