#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

set -eu

echo "Running on runner_local"
PBS=./examples/semeval2022task10/seq2seq_parser/scripts/crosslingual_zeroshot/runner_local.pbs


for SEED in {0..2} ; do
  PLM=./log/semeval2022task10/seq2seq_monolingual/opener_en-google-mt5-large-${SEED}
  echo "Run seed=${SEED}, plm='${PLM}' ${PBS}"
  export PLM=${PLM} SEED=${SEED} 
  ${PBS}
  sleep 5
done

