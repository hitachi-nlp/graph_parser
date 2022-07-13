#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

# Need to export
# LOG_ROOT
# CORPORA
# TGT_LANG
# SEED
# PLM

set -eu

CORPORA=($(echo $CORPORA | tr ":" "\n"))
echo "Corpora: ${CORPORA[@]}"
export TOKENIZERS_PARALLELISM=false

mkdir -p ${LOG_ROOT}

if [ ! -e ${LOG_ROOT}/train.mrp ]; then
  TRAIN_PATHS=()
  for CORPUS in ${CORPORA[@]}; do
      TRAIN_PATHS+=("data/input/${CORPUS}_train.${TGT_LANG}.mrp")
  done
  cat ${TRAIN_PATHS[@]} > ${LOG_ROOT}/train.mrp
fi

export FTRAIN=${LOG_ROOT}/train.mrp
export FDEV=${FTRAIN}  # Workaround

if [ ! -e ${LOG_ROOT}/data_statistics.log ]; then
  echo "------ Statistics of train data ------- : " >> ${LOG_ROOT}/data_statistics.log 2>&1
  echo "Train corpora: ${CORPORA[@]}" >> ${LOG_ROOT}/data_statistics.log 2>&1
  echo "Train data paths: ${TRAIN_PATHS[@]}" >> ${LOG_ROOT}/data_statistics.log 2>&1
  python -m utils.show_label_stats --input ${FTRAIN} >> ${LOG_ROOT}/data_statistics.log 2>&1
  mtool --read mrp --analyze ${FTRAIN} >> ${LOG_ROOT}/data_statistics.log 2>&1
fi

# Train
if [ ! -e ${LOG_ROOT}/finished.json ]; then
  export SEED=${SEED}
  BS=16
  N=$(cat ${FTRAIN} | wc -l)
  export EPOCHS=$(python -c "print(int(10000/(${N}/${BS})))")
  python -m amparse.trainer.train \
    --log ${LOG_ROOT} \
    --ftrain ${FTRAIN} \
    --fvalid ${FDEV} \
    --seed ${SEED} \
    --model_name_or_path ${PLM} \
    --build_numericalizer_on_entire_corpus true \
    --split_document false \
    --max_encode 512 \
    --batch_size 16 \
    --eval_batch_size 16 \
    --postprocessor "ssa:ssa,opener_en:ssa,opener_es:ssa,norec:ssa,multibooked_ca:ssa,multibooked_eu:ssa,darmstadt_unis:ssa,mpqa:ssa" \
    --embed_dropout 0.1 \
    --mlp_dropout 0.1 \
    --dim_mlp 512 \
    --dim_biaffine 512 \
    --lambda_bio 1. \
    --lambda_proposition 0. \
    --lambda_arc 1. \
    --lambda_rel 0.1 \
    --lambda_tgt_fw 1.0 \
    --lambda_other_fw 1.0 \
    --lr 2e-5 \
    --beta1 0.9 \
    --beta2 0.998 \
    --warmup_ratio 0.1 \
    --clip 1.0 \
    --epochs ${EPOCHS} \
    --terminate_epochs ${EPOCHS} \
    --evaluate_epochs ${EPOCHS} \
    --disable_saving_large_files false \
    --disable_evaluation true \
    --evaluate_with_oracle_span false

  rm -r ${LOG_ROOT}/evaluation
fi

for CORPUS in ${CORPORA[@]}
do
  # Prediction and evaluation for dev data
  if [ ! -e ${LOG_ROOT}/dev/eval.txt ]; then
    export LOG_DIR=${LOG_ROOT}/dev
    python -m amparse.predictor.predict \
        --log ${LOG_DIR} \
        --models ${LOG_ROOT}/model \
        --input data/input/${CORPUS}_dev.${TGT_LANG}.mrp \
        --batch_size 16 \
        --image false \
        --oracle_span false
    python -m utils.converter.mrp2ssa --input ${LOG_DIR}/prediction.mrp --output ${LOG_DIR}/predictions.json
    python data/orig/semeval22_structured_sentiment/evaluation/evaluate_single_dataset.py \
      data/orig/semeval22_structured_sentiment/data/${CORPUS}/dev.json ${LOG_DIR}/predictions.json > ${LOG_DIR}/eval.txt
  fi

  # Prediction for test data
  if [ ! -e ${LOG_ROOT}/test/eval.txt ]; then
    export LOG_DIR=${LOG_ROOT}/test
    python -m amparse.predictor.predict \
        --log ${LOG_DIR} \
        --models ${LOG_ROOT}/model \
        --input data/input/${CORPUS}_test.${TGT_LANG}.mrp\
        --batch_size 16 \
        --image false \
        --oracle_span false
    python -m utils.converter.mrp2ssa --input ${LOG_DIR}/prediction.mrp --output ${LOG_DIR}/predictions.json
    python data/orig/semeval22_structured_sentiment/evaluation/evaluate_single_dataset.py \
      data/orig/semeval22_structured_sentiment/data/${CORPUS}/test.json ${LOG_DIR}/predictions.json > ${LOG_DIR}/eval.txt
  fi
done
