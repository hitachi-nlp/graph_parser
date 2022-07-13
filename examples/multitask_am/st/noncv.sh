#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


set -eu

export TOKENIZERS_PARALLELISM=false

for SEED in {0..29} ; do
    SEED_DIR=./log/multitask_am/st/${CORPUS}/${SEED}
    if [ -e ${SEED_DIR}/finetune/finished.json ]; then
        echo "${SEED_DIR} already exists."
        continue
    fi

    mkdir -p ${SEED_DIR}/data

    cp ./examples/multitask_am/tuned_hyperparameters/${CORPUS}.sh ${SEED_DIR}/hparam.sh
    chmod 755 ${SEED_DIR}/hparam.sh

    cp data/input/${CORPUS}_train.mrp ${SEED_DIR}/data/train.mrp
    cp data/input/${CORPUS}_dev.mrp ${SEED_DIR}/data/dev.mrp
    cp data/input/${CORPUS}_test.mrp ${SEED_DIR}/data/test.mrp

    python -m utils.sample_mrp \
      --input ${SEED_DIR}/data/train.mrp \
      --output ${SEED_DIR}/data/train.mrp \
      --output_rate ${TRAIN_AMOUNT} \
      --min_output 1 \
      --seed ${SEED}
    python -m utils.sample_mrp \
      --input ${SEED_DIR}/data/dev.mrp \
      --output ${SEED_DIR}/data/dev.mrp \
      --output_rate ${TRAIN_AMOUNT} \
      --min_output 1 \
      --seed ${SEED}

    python -m amparse.common.validate_mrp \
      --group1 ${SEED_DIR}/data/train.mrp ${SEED_DIR}/data/dev.mrp \
      --group2 ${SEED_DIR}/data/test.mrp \
      --way exclusive >> ${SEED_DIR}/data/data_validation.log 2>&1

    echo "------ Statistics of train data ------- : " >> ${SEED_DIR}/data/data_statistics.log 2>&1
    python -m utils.show_label_stats --input ${SEED_DIR}/data/train.mrp >> ${SEED_DIR}/data/data_statistics.log 2>&1
    mtool --read mrp --analyze ${SEED_DIR}/data/train.mrp >> ${SEED_DIR}/data/data_statistics.log 2>&1
    echo "------ Statistics of dev data ------- : " >> ${SEED_DIR}/data/data_statistics.log 2>&1
    python -m utils.show_label_stats --input ${SEED_DIR}/data/dev.mrp >> ${SEED_DIR}/data/data_statistics.log 2>&1
    mtool --read mrp --analyze ${SEED_DIR}/data/dev.mrp >> ${SEED_DIR}/data/data_statistics.log 2>&1
    echo "------ Statistics of test data ------- : " >> ${SEED_DIR}/data/data_statistics.log 2>&1
    python -m utils.show_label_stats --input ${SEED_DIR}/data/test.mrp >> ${SEED_DIR}/data/data_statistics.log 2>&1
    mtool --read mrp --analyze ${SEED_DIR}/data/test.mrp >> ${SEED_DIR}/data/data_statistics.log 2>&1

    # Target corpus fine-tuning
    source ${SEED_DIR}/hparam.sh

    python -m amparse.trainer.train \
      --ftrain ${SEED_DIR}/data/train.mrp \
      --fvalid ${SEED_DIR}/data/dev.mrp \
      --ftest ${SEED_DIR}/data/test.mrp \
      --seed ${SEED} \
      --model_name_or_path "allenai/longformer-base-4096" \
      --log ${SEED_DIR}/finetune \
      --build_numericalizer_on_entire_corpus \
      --split_document false \
      --attention_window 512 \
      --batch_size 4 \
      --eval_batch_size 16 \
      --postprocessor "default:default,aaec:aaec,aaec_essay:aaec,aaec_para:aaec,mtc:mtc,cdcp:cdcp,abstrct:abstrct,aasd:aasd,tree:mtc,trees:aaec,graph:cdcp" \
      --embed_dropout 0.1 \
      --mlp_dropout 0.1 \
      --dim_mlp 768 \
      --dim_biaffine 768 \
      --lambda_bio 1.0 \
      --lambda_proposition ${lambda_proposition} \
      --lambda_arc ${lambda_arc} \
      --lambda_rel ${lambda_rel} \
      --lambda_tgt_fw 1.0 \
      --lambda_other_fw 1.0 \
      --lr ${lr} \
      --beta1 0.9 \
      --beta2 0.998 \
      --warmup_ratio 0.1 \
      --clip 5.0 \
      --epochs 20 \
      --terminate_epochs 20 \
      --evaluate_epochs 2
    # -> The trained model was saved into "${SEED_DIR}/finetune/model"

    # Prediction with oracle span
    export LOG_DIR=${SEED_DIR}/finetune/os
    python -m amparse.predictor.predict \
      --log ${LOG_DIR} \
      --models ${SEED_DIR}/finetune/model \
      --input ${SEED_DIR}/data/test.mrp \
      --batch_size 16 \
      --oracle_span

    mkdir -p ${LOG_DIR}/prediction/
    mkdir -p ${LOG_DIR}/evaluation/
    mv ${LOG_DIR}/prediction.mrp ${LOG_DIR}/prediction/test.mrp
    python amparse/evaluator/scorer.py \
        -system ${LOG_DIR}/prediction/test.mrp \
        -gold ${SEED_DIR}/data/test.mrp >> ${LOG_DIR}/evaluation/test.jsonl
done
