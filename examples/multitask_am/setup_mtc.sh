#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


SAVE_DIR=data/input

# Preprocess MTC dataset
mkdir -p data/orig/mtc
wget --no-check-certificate https://github.com/peldszus/arg-microtexts/archive/refs/heads/master.zip
mv master.zip arg-microtexts-master.zip
mv arg-microtexts-master.zip data/orig/mtc/
rm -rf arg-microtexts-master/
unzip data/orig/mtc/arg-microtexts-master.zip
rm -rf data/mtc/
mkdir -p data/mtc/
mv arg-microtexts-master data/mtc/

SEED=42

python -m utils.converter.mtc2mrp \
  --dir_mtc data/mtc/arg-microtexts-master/corpus/en \
  --prefix MTC_ \
  --seed ${SEED} \
  --output_cv_prefix ${SAVE_DIR}/cv/mtc \
  --output ${SAVE_DIR}/mtc.mrp \
  --dev_rate 0.1 \
  --language "en"

rm -rf data/mtc/
