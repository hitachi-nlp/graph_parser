#! /bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

set -eu

ROOT_DIR=./examples/semeval2022task10
DATA_DIR=./data/orig/semeval22_structured_sentiment/data
OUT_DIR=./data/input

for SPLIT in train dev test
do
  for CORPUS in darmstadt_unis mpqa multibooked_ca multibooked_eu norec opener_en opener_es
  do
    FREAD=${DATA_DIR}/${CORPUS}/${SPLIT}.json
    SAVEPREFIX=${OUT_DIR}/${CORPUS}_${SPLIT}
    echo "${FREAD} -> ${SAVEPREFIX}"
    python ${ROOT_DIR}/seq2seq_parser/to_ssa.py $FREAD > ${SAVEPREFIX}.ssa
    python ${ROOT_DIR}/seq2seq_parser/to_ssa_linear.py ${SAVEPREFIX}.ssa --no-check-span --no-skip-empty > ${SAVEPREFIX}.ssal.json
  done
done

for SPLIT in train dev
do
  cat ${OUT_DIR}/opener_en_${SPLIT}.ssal.json \
    ${OUT_DIR}/mpqa_${SPLIT}.ssal.json \
    ${OUT_DIR}/darmstadt_unis_${SPLIT}.ssal.json > ${OUT_DIR}/opener_en_mpqa_darmstadt_unis_${SPLIT}.ssal.json
done

