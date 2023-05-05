#!/usr/bin/env bash
# bash prepare-docbpe.sh data-cadec/sent data-cadec/mix exp_test/cadec.tokenized.en-de

if [ -d "mosesdecoder" ]; then
    echo "mosesdecoder already exists, skipping download"
else
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git
fi

if [ -d "subword-nmt" ]; then
    echo "subword-nmt already exists, skipping download"
else
    echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
    git clone https://github.com/rsennrich/subword-nmt.git
fi

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
BPEROOT=subword-nmt/subword_nmt

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

sent=$1
data=$2
prep=$3

src=en
tgt=ru
lang=$src-$tgt
tmp=$prep/tmp

BPE_TOKENS=30000
BPE_CODE=$prep/code
if [ -f "$BPE_CODE" ]; then
    echo "BPE code $BPE_CODE is already exist, skipping data preparation."
    exit
fi

mkdir -p $tmp $prep


echo "filter out empty lines from original data and split doc with empty line..."
for D in sent train dev test; do
    if [ $D == "sent" ]; then
        sf=$sent/concatenated_${src}2${tgt}_train_${src}.txt
        tf=$sent/concatenated_${src}2${tgt}_train_${tgt}.txt
    else
        sf=$data/concatenated_${src}2${tgt}_${D}_${src}.txt
        tf=$data/concatenated_${src}2${tgt}_${D}_${tgt}.txt
    fi

    RD=$D
    if [ $D == "dev" ]; then
        RD=valid
    fi

    rf=$tmp/$RD.$lang.tag
    echo $rf

    paste -d"\t" $sf $tf | \
    grep -v -P "^\s*\t" | \
    grep -v -P "\t\s*$" | \
    sed -e 's/\r//g' > $rf

    cut -f 1 $rf | \
    sed -e 's/^<d>\s*$//g' | \
    perl $TOKENIZER -threads 8 -l $src | \
    perl $LC > $tmp/$RD.$src

    cut -f 2 $rf | \
    sed -e 's/^<d>\s*$//g' | \
    perl $TOKENIZER -threads 8 -l $tgt | \
    perl $LC > $tmp/$RD.$tgt
done

echo "learn_bpe.py on ${TRAIN}..."
TRAIN=$tmp/train.$lang
cat $tmp/sent.$src > $TRAIN
cat $tmp/sent.$tgt >> $TRAIN

python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for F in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${F}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$F > $tmp/$F.bpe
    done
done

echo "apply doc-level special tags..."
for L in $src $tgt; do
    for F in train.$L valid.$L test.$L; do
        cat $tmp/$F.bpe | \
        # replace empty line with [DOC]
        sed -e 's/^$/[DOC]/g' | \
        # connect all lines into one line
        sed -z -e 's/\n/ [SEP] /g' | \
        # replace the begin of doc with newline
        sed -e 's/ \[DOC\] \[SEP\] /\n/g' | \
        # handle the begin-symbol of the first doc
        sed -e 's/\[DOC\] \[SEP\] //g' | \
        # replace all [SEP] with </s>
        sed -e 's/\[SEP\]/<\/s>/g' > $prep/$F
    done
done
