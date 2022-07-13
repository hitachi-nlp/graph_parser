#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


set -eu

# Validate mrp files
DATA_DIR=data/input
for PREFIX in aaec_essay aaec_para abstrct cdcp
do
  python -m amparse.common.validate_mrp \
    --group1 ${DATA_DIR}/${PREFIX}.mrp
  python -m amparse.common.validate_mrp \
    --group1 ${DATA_DIR}/${PREFIX}_train.mrp ${DATA_DIR}/${PREFIX}_dev.mrp \
    --group2 ${DATA_DIR}/${PREFIX}_test.mrp \
    --way exclusive
  python -m amparse.common.validate_mrp \
    --group1 ${DATA_DIR}/${PREFIX}_train_dev.mrp \
    --group2 ${DATA_DIR}/${PREFIX}_train.mrp ${DATA_DIR}/${PREFIX}_dev.mrp \
    --way inclusive
  python -m amparse.common.validate_mrp \
    --group1 ${DATA_DIR}/${PREFIX}_train_dev.mrp \
    --group2 ${DATA_DIR}/${PREFIX}_test.mrp \
    --way exclusive
done

# Validate CV mrp files
for PREFIX in aasd mtc
do
  for CV in {0..49}
  do
    python -m amparse.common.validate_mrp \
      --group1 ${DATA_DIR}/cv/${PREFIX}.cv${CV}.train.mrp ${DATA_DIR}/cv/${PREFIX}.cv${CV}.dev.mrp \
      --group2 ${DATA_DIR}/cv/${PREFIX}.cv${CV}.test.mrp \
      --way exclusive
    python -m amparse.common.validate_mrp \
      --group1 ${DATA_DIR}/cv/${PREFIX}.cv${CV}.train.mrp ${DATA_DIR}/cv/${PREFIX}.cv${CV}.dev.mrp ${DATA_DIR}/cv/${PREFIX}.cv${CV}.test.mrp \
      --group2 ${DATA_DIR}/${PREFIX}.mrp \
      --way inclusive
  done
done

# We generated hash for the input file by "md5sum data/input/*.mrp data/input/**/*.mrp > examples/multitask_am/hash.md5"
# To use hash check on mac, install following "brew install md5sha1sum"
md5sum -c examples/multitask_am/hash.md5
