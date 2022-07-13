#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

set -eu


CORPUS="opener_es"
PLM="google/mt5-large"

echo "Running on runner_local"

PBS=./examples/semeval2022task10/seq2seq_parser/scripts/runner_local.pbs


for SEED in {0..2} ; do
  echo "Run seed=${SEED}, plm='${PLM}', ${PBS}"
  export CORPUS=${CORPUS} PLM=${PLM} SEED=${SEED} 
  ${PBS}
  sleep 5
done

