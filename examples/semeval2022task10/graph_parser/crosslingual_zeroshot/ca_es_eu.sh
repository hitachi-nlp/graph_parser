#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

set -eu

SRC_CORPORA="opener_en"
TGT_CORPORA="multibooked_ca:multibooked_eu:opener_es"
TGT_LANGS="ca:es:eu"
SCRIPT=./examples/semeval2022task10/graph_parser/crosslingual_zeroshot/runner.sh

echo "Running on runner_local"
PBS=./examples/semeval2022task10/graph_parser/crosslingual_zeroshot/runner_local.pbs


for SEED in {0..2} ; do
  LOG_ROOT=./log/semeval2022task10/graph_crosslingual_zeroshot/${SEED}
  echo "Run seed=${SEED}, source_corpora='${SRC_CORPORA}' logging_root=${LOG_ROOT} target_langs=${TGT_LANGS} target_corpora='${TGT_CORPORA}' ${SCRIPT}"
  export TGT_CORPORA=${TGT_CORPORA} SRC_CORPORA=${SRC_CORPORA} LOG_ROOT=${LOG_ROOT} TGT_LANGS=${TGT_LANGS} SCRIPT=${SCRIPT} SEED=${SEED}
  ${PBS}
  sleep 5
done

ENSEMBLE_SCRIPT=./examples/semeval2022task10/graph_parser/crosslingual_zeroshot/runner_ensemble.sh
MODEL_ROOT=./log/semeval2022task10/graph_crosslingual_zeroshot/
export TGT_CORPORA=${TGT_CORPORA} SRC_CORPORA=${SRC_CORPORA} MODEL_ROOT=${MODEL_ROOT} TGT_LANGS=${TGT_LANGS} SCRIPT=${ENSEMBLE_SCRIPT} 
${PBS}

