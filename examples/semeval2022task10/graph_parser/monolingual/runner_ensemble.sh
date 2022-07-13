#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

# Need to export
# MODEL_ROOT
# CORPORA
# TGT_LANG

set -eu

export TOKENIZERS_PARALLELISM=false

CORPORA=($(echo $CORPORA | tr ":" "\n"))

for SEED in {0..2}
do
  while [ ! -e ${MODEL_ROOT}/${SEED}/model ]
  do
    sleep 10
  done
done

sleep 10

for CORPUS in ${CORPORA[@]}
do
  ENSEMBLE_DIR=${MODEL_ROOT}/ensemble

  # Prediction and evaluation for dev data
  export LOG_DIR=${ENSEMBLE_DIR}/dev
  if [ ! -e ${ENSEMBLE_DIR}/dev/eval.txt ]; then
    python -m amparse.predictor.predict \
        --log ${LOG_DIR} \
        --models ${MODEL_ROOT}/0/model ${MODEL_ROOT}/1/model ${MODEL_ROOT}/2/model \
        --input data/input/${CORPUS}_dev.${TGT_LANG}.mrp \
        --batch_size 16 \
        --image false \
        --oracle_span false
    python -m utils.converter.mrp2ssa --input ${LOG_DIR}/prediction.mrp --output ${LOG_DIR}/predictions.json
    python data/orig/semeval22_structured_sentiment/evaluation/evaluate_single_dataset.py \
      data/orig/semeval22_structured_sentiment/data/${CORPUS}/dev.json ${LOG_DIR}/predictions.json > ${LOG_DIR}/eval.txt
  fi
  if [ ! -d ${ENSEMBLE_DIR}/dev/images ]; then
      python -m utils.visualize_graphs --input ${LOG_DIR}/prediction.mrp --log ${ENSEMBLE_DIR}/dev/images --formats pdf --cores 4
  fi

  # Prediction for test data
  export LOG_DIR=${ENSEMBLE_DIR}/test
  if [ ! -e ${ENSEMBLE_DIR}/test/eval.txt ]; then
    python -m amparse.predictor.predict \
        --log ${LOG_DIR} \
        --models ${MODEL_ROOT}/0/model ${MODEL_ROOT}/1/model ${MODEL_ROOT}/2/model \
        --input data/input/${CORPUS}_test.${TGT_LANG}.mrp \
        --batch_size 16 \
        --image false \
        --oracle_span false
    python -m utils.converter.mrp2ssa --input ${LOG_DIR}/prediction.mrp --output ${LOG_DIR}/predictions.json
    python data/orig/semeval22_structured_sentiment/evaluation/evaluate_single_dataset.py \
      data/orig/semeval22_structured_sentiment/data/${CORPUS}/test.json ${LOG_DIR}/predictions.json > ${LOG_DIR}/eval.txt
  fi
  if [ ! -d ${ENSEMBLE_DIR}/test/images ]; then
      python -m utils.visualize_graphs --input ${LOG_DIR}/prediction.mrp --log ${ENSEMBLE_DIR}/test/images --formats pdf --cores 4
  fi
done
