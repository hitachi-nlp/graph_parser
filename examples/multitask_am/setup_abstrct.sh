#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


SAVE_DIR=data/input

# Preprocess AbstRCT dataset
mkdir -p data/orig/abstrct
wget --no-check-certificate https://gitlab.com/tomaye/abstrct/-/archive/master/abstrct-master.zip
mv abstrct-master.zip data/orig/abstrct/
unzip data/orig/abstrct/abstrct-master.zip
rm -rf data/abstrct/
mkdir -p data/abstrct/
mv abstrct-master data/abstrct/

python -m utils.converter.abstrct2mrp \
  --dir_abstrct data/abstrct/abstrct-master/AbstRCT_corpus/data/train/neoplasm_train \
  --prefix AbstRCT_ \
  --output ${SAVE_DIR}/abstrct_train.mrp

python -m utils.converter.abstrct2mrp \
  --dir_abstrct data/abstrct/abstrct-master/AbstRCT_corpus/data/dev/neoplasm_dev \
  --prefix AbstRCT_ \
  --output ${SAVE_DIR}/abstrct_dev.mrp

python -m utils.converter.abstrct2mrp \
  --dir_abstrct data/abstrct/abstrct-master/AbstRCT_corpus/data/test/neoplasm_test \
  --prefix AbstRCT_ \
  --output ${SAVE_DIR}/abstrct_test.mrp

cat ${SAVE_DIR}/abstrct_train.mrp ${SAVE_DIR}/abstrct_dev.mrp > ${SAVE_DIR}/abstrct_train_dev.mrp
cat ${SAVE_DIR}/abstrct_train.mrp ${SAVE_DIR}/abstrct_dev.mrp ${SAVE_DIR}/abstrct_test.mrp > ${SAVE_DIR}/abstrct.mrp

rm -rf data/abstrct/

