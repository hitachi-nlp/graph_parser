#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


SAVE_DIR=data/input

# Preprocess AASD dataset
mkdir -p data/orig/aasd
wget --no-check-certificate http://scientmin.taln.upf.edu/argmin/scidtb_argmin_annotations.tgz
mv scidtb_argmin_annotations.tgz data/orig/aasd/
rm -rf data/aasd/
mkdir -p data/aasd/
tar -xvzf data/orig/aasd/scidtb_argmin_annotations.tgz -C data/aasd/

python -m utils.converter.aasd2mrp \
  --dir_aasd data/aasd/ \
  --prefix AASD_ \
  --output ${SAVE_DIR}/aasd.mrp

python -m utils.split_mrp_for_cv \
  --input ${SAVE_DIR}/aasd.mrp \
  --dir_output ${SAVE_DIR}/cv/ \
  --seed 42 \
  --dev_rate 0.1 \
  --iter 10 \
  --fold 5

rm -rf data/aasd/
