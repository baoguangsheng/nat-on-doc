#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
pip install protobuf==3.19.1
pip install ninja==1.11.1 sacrebleu==1.4.14 tensorboard==2.6.0

pip install ./GLAT/imputer-pytorch/.
user_site=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
cp ./GLAT/imputer-pytorch/torch_imputer/*.* $user_site/torch_imputer/

pip install --editable ./GLAT/.
cp ./plugins_gtrans/generate.py ./GLAT/fairseq_cli/generate2.py -f
cp ./plugins_gtrans/validate.py ./GLAT/fairseq_cli/validate2.py -f
