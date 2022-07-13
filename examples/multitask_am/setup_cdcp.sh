#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


SAVE_DIR=data/input

# Preprocess CDCP dataset
mkdir -p data/orig/cdcp
wget --no-check-certificate https://facultystaff.richmond.edu/~jpark/data/cdcp_acl17.zip
mv cdcp_acl17.zip data/orig/cdcp/

rm -rf cdcp_acl17/
unzip data/orig/cdcp/cdcp_acl17.zip
rm -rf data/cdcp/
mkdir -p data/cdcp/cdcp_acl17
mv cdcp/train data/cdcp/cdcp_acl17
mv cdcp/test data/cdcp/cdcp_acl17

python -m utils.converter.cdcp2mrp \
  --dir_cdcp data/cdcp/cdcp_acl17/train \
  --prefix CDCP_ \
  --output ${SAVE_DIR}/cdcp_train.mrp

python -m utils.split_mrp \
  --input ${SAVE_DIR}/cdcp_train.mrp \
  --output1 ${SAVE_DIR}/cdcp_train.mrp \
  --output2 ${SAVE_DIR}/cdcp_dev.mrp \
  --output2_rate 0.1 \
  --seed 42

python -m utils.converter.cdcp2mrp \
--dir_cdcp data/cdcp/cdcp_acl17/test \
--prefix CDCP_ \
--output ${SAVE_DIR}/cdcp_test.mrp

cat ${SAVE_DIR}/cdcp_train.mrp ${SAVE_DIR}/cdcp_dev.mrp > ${SAVE_DIR}/cdcp_train_dev.mrp
cat ${SAVE_DIR}/cdcp_train.mrp ${SAVE_DIR}/cdcp_dev.mrp ${SAVE_DIR}/cdcp_test.mrp > ${SAVE_DIR}/cdcp.mrp

rm -rf cdcp/
rm -rf data/cdcp/
rm -rf ta/
rm -rf __MACOSX/
rm readme.txt
