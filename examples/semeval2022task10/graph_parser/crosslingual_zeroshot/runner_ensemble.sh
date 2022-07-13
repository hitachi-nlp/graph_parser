#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

# Need to export
# TGT_CORPORA
# SRC_CORPORA
# TGT_LANGS
# MODEL_ROOT

set -eu

export TOKENIZERS_PARALLELISM=false

SRC_CORPORA=($(echo $SRC_CORPORA | tr ":" "\n"))
TGT_CORPORA=($(echo $TGT_CORPORA | tr ":" "\n"))
TGT_LANGS=($(echo $TGT_LANGS | tr ":" "\n"))

for SEED in {0..2}
do
  while [ ! -e ${MODEL_ROOT}/${SEED}/model ]
  do
    sleep 10
  done
done

for TGT_LANG in ${TGT_LANGS[@]}
do
  for CORPUS in ${TGT_CORPORA[@]}
  do
    if [ ! -e data/input/${CORPUS}_dev.${TGT_LANG}.mrp ]; then
      continue
    fi

    ENSEMBLE_DIR=${MODEL_ROOT}/ensemble/${CORPUS}

    # Zeros-shot for dev data
    export LOG_DIR=${ENSEMBLE_DIR}/dev
    if [ ! -e ${ENSEMBLE_DIR}/dev/eval.txt ]; then
      python -m amparse.predictor.predict \
          --log ${LOG_DIR} \
          --models ${MODEL_ROOT}/0/model ${MODEL_ROOT}/1/model ${MODEL_ROOT}/2/model \
          --input data/input/${CORPUS}_dev.${TGT_LANG}.mrp \
          --batch_size 16 \
          --image false \
          --overwrite_framework "${SRC_CORPORA[0]}" \
          --oracle_span false
      python -m utils.converter.mrp2ssa --input ${LOG_DIR}/prediction.mrp --output ${LOG_DIR}/predictions.json
      python data/orig/semeval22_structured_sentiment/evaluation/evaluate_single_dataset.py \
        data/orig/semeval22_structured_sentiment/data/${CORPUS}/dev.json ${LOG_DIR}/predictions.json > ${LOG_DIR}/eval.txt
    fi
    if [ ! -d ${ENSEMBLE_DIR}/dev/images ]; then
        python -m utils.visualize_graphs --input ${LOG_DIR}/prediction.mrp --log ${ENSEMBLE_DIR}/dev/images --formats pdf --cores 4
    fi

    # Zeros-shot for test data
    export LOG_DIR=${ENSEMBLE_DIR}/test
    if [ ! -e ${ENSEMBLE_DIR}/test/eval.txt ]; then
      python -m amparse.predictor.predict \
          --log ${LOG_DIR} \
          --models ${MODEL_ROOT}/0/model ${MODEL_ROOT}/1/model ${MODEL_ROOT}/2/model \
          --input data/input/${CORPUS}_test.${TGT_LANG}.mrp \
          --batch_size 16 \
          --image false \
          --overwrite_framework "${SRC_CORPORA[0]}" \
          --oracle_span false
      python -m utils.converter.mrp2ssa --input ${LOG_DIR}/prediction.mrp --output ${LOG_DIR}/predictions.json
      python data/orig/semeval22_structured_sentiment/evaluation/evaluate_single_dataset.py \
        data/orig/semeval22_structured_sentiment/data/${CORPUS}/test.json ${LOG_DIR}/predictions.json > ${LOG_DIR}/eval.txt
    fi
    if [ ! -d ${ENSEMBLE_DIR}/test/images ]; then
        python -m utils.visualize_graphs --input ${LOG_DIR}/prediction.mrp --log ${ENSEMBLE_DIR}/test/images --formats pdf --cores 4
    fi
  done
done
