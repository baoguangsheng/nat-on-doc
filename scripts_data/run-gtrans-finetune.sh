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

# switch to submodule and convert the relative path to absolute
cur_dir=$(pwd)
exp_path=$cur_dir/$exp_path
cd ./G-Trans

if [ $mode == "data" ]; then
  bash -e exp_gtrans/prepare-finetune.sh $data $exp_path

elif [ $mode == "train" ]; then
  bash -e exp_gtrans/run-finetune.sh $data train $exp_path

elif [ $mode == "test" ]; then
  bash -e exp_gtrans/run-finetune.sh $data test $exp_path

else
  echo Unknown mode ${mode}.
fi

cd $cur_dir