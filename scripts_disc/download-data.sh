#!/usr/bin/env bash
# bash prepare-docdata.sh data-cadec
data_path=$1

mkdir -p $data_path

echo `date`, downloading CADec data ...
#wget https://www.dropbox.com/s/5drjpx07541eqst/acl19_good_translation_wrong_in_context.zip
unzip acl19_good_translation_wrong_in_context.zip -d $data_path/orig
cp $data_path/orig/data_to_publish/* $data_path/orig/. -r

echo `date`, downloading scoring data ...
cp CADec/consistency_testsets/scoring_data $data_path/orig/scoring_data -r

echo `date`, converting data ...
python -m scripts_disc.data_converter --datadir $data_path/orig  --destdir $data_path
