#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


set -eu

export TOKENIZERS_PARALLELISM=false

for CV in {0..49} ; do
    CV_DIR=./log/multitask_am/st/${CORPUS}/${CV}
    if [ -e ${CV_DIR}/finetune/finished.json ]; then
	echo "${CV_DIR} already exists."
        continue
    fi

    mkdir -p ${CV_DIR}/data

    cp ./examples/multitask_am/tuned_hyperparameters/${CORPUS}.sh ${CV_DIR}/hparam.sh
    chmod 755 ${CV_DIR}/hparam.sh

    cp data/input/cv/${CORPUS}.cv${CV}.train.mrp ${CV_DIR}/data/train.mrp
    cp data/input/cv/${CORPUS}.cv${CV}.dev.mrp ${CV_DIR}/data/dev.mrp
    cp data/input/cv/${CORPUS}.cv${CV}.test.mrp ${CV_DIR}/data/test.mrp

    python -m utils.sample_mrp \
      --input ${CV_DIR}/data/train.mrp \
      --output ${CV_DIR}/data/train.mrp \
      --output_rate ${TRAIN_AMOUNT} \
      --min_output 1 \
      --seed 0
    python -m utils.sample_mrp \
      --input ${CV_DIR}/data/dev.mrp \
      --output ${CV_DIR}/data/dev.mrp \
      --output_rate ${TRAIN_AMOUNT} \
      --min_output 1 \
      --seed 0

    python -m amparse.common.validate_mrp \
      --group1 ${CV_DIR}/data/train.mrp ${CV_DIR}/data/dev.mrp \
      --group2 ${CV_DIR}/data/test.mrp \
      --way exclusive >> ${CV_DIR}/data/data_validation.log 2>&1

    echo "------ Statistics of train data ------- : " >> ${CV_DIR}/data/data_statistics.log 2>&1
    python -m utils.show_label_stats --input ${CV_DIR}/data/train.mrp >> ${CV_DIR}/data/data_statistics.log 2>&1
    mtool --read mrp --analyze ${CV_DIR}/data/train.mrp >> ${CV_DIR}/data/data_statistics.log 2>&1
    echo "------ Statistics of dev data ------- : " >> ${CV_DIR}/data/data_statistics.log 2>&1
    python -m utils.show_label_stats --input ${CV_DIR}/data/dev.mrp >> ${CV_DIR}/data/data_statistics.log 2>&1
    mtool --read mrp --analyze ${CV_DIR}/data/dev.mrp >> ${CV_DIR}/data/data_statistics.log 2>&1
    echo "------ Statistics of test data ------- : " >> ${CV_DIR}/data/data_statistics.log 2>&1
    python -m utils.show_label_stats --input ${CV_DIR}/data/test.mrp >> ${CV_DIR}/data/data_statistics.log 2>&1
    mtool --read mrp --analyze ${CV_DIR}/data/test.mrp >> ${CV_DIR}/data/data_statistics.log 2>&1

    # Target corpus fine-tuning
    source ${CV_DIR}/hparam.sh

    python -m amparse.trainer.train \
      --ftrain ${CV_DIR}/data/train.mrp \
      --fvalid ${CV_DIR}/data/dev.mrp \
      --ftest ${CV_DIR}/data/test.mrp \
      --seed 0 \
      --model_name_or_path "allenai/longformer-base-4096" \
      --log ${CV_DIR}/finetune \
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
    # -> The trained model was saved into "${CV_DIR}/finetune/model"

    # Prediction with oracle span
    LOG_DIR=${CV_DIR}/finetune/os
    python -m amparse.predictor.predict \
      --log ${LOG_DIR} \
      --models ${CV_DIR}/finetune/model \
      --input ${CV_DIR}/data/test.mrp \
      --batch_size 16 \
      --oracle_span

    mkdir -p ${LOG_DIR}/prediction/
    mkdir -p ${LOG_DIR}/evaluation/
    mv ${LOG_DIR}/prediction.mrp ${LOG_DIR}/prediction/test.mrp
    python amparse/evaluator/scorer.py \
        -system ${LOG_DIR}/prediction/test.mrp \
        -gold ${CV_DIR}/data/test.mrp >> ${LOG_DIR}/evaluation/test.jsonl
done
