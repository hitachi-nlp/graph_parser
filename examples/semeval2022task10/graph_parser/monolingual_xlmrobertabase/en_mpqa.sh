#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

set -eu

CORPORA="mpqa"
TGT_LANG="en"
PLM="xlm-roberta-base"
SCRIPT=./examples/semeval2022task10/graph_parser/monolingual_xlmrobertabase/runner.sh

echo "Running on runner_local"
PBS=./examples/semeval2022task10/graph_parser/monolingual_xlmrobertabase/runner_local.pbs


for SEED in {0..2} ; do
  LOG_ROOT=./log/semeval2022task10/graph_monolingual_xlmrobertabase/${TGT_LANG}_${CORPORA}/${SEED}
  echo "Run seed=${SEED}, corpora='${CORPORA}' lang=${TGT_LANG} plm=${PLM} logging_dir=${LOG_ROOT}, ${SCRIPT}"
  export CORPORA=${CORPORA} TGT_LANG=${TGT_LANG} PLM=${PLM} SCRIPT=${SCRIPT} SEED=${SEED} LOG_ROOT=${LOG_ROOT} 
  ${PBS}
  sleep 5
done

ENSEMBLE_SCRIPT=./examples/semeval2022task10/graph_parser/monolingual_xlmrobertabase/runner_ensemble.sh
MODEL_ROOT=./log/semeval2022task10/graph_monolingual_xlmrobertabase/${TGT_LANG}_${CORPORA}
export CORPORA=${CORPORA} TGT_LANG=${TGT_LANG} MODEL_ROOT=${MODEL_ROOT} SCRIPT=${ENSEMBLE_SCRIPT} 
${PBS}