#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

for SPLIT in train dev test
do
  for DATA_PREFIX in darmstadt_unis_${SPLIT}.en mpqa_${SPLIT}.en opener_en_${SPLIT}.en opener_es_${SPLIT}.es multibooked_ca_${SPLIT}.ca multibooked_eu_${SPLIT}.eu norec_${SPLIT}.no
  do
    OUT_DIR=log/semeval2022task10/gold_graphs
    python -m utils.visualize_graphs --input data/input/${DATA_PREFIX}.mrp --log ${OUT_DIR}/${DATA_PREFIX} --formats pdf --cores 4
  done
done