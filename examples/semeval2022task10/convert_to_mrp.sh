#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

#set -eu

SAVE_DIR=data/input

# norec
PREFIX=norec
LANGUAGE=no
for SPLIT in train dev test
do
  SAVE_PATH=${SAVE_DIR}/${PREFIX}_${SPLIT}.${LANGUAGE}.mrp
  if [ -e ${SAVE_PATH} ]; then
    echo "${SAVE_PATH} already exists. Skipping."
    continue
  fi
  python -m utils.converter.ssa2mrp \
    --input ./data/orig/semeval22_structured_sentiment/data/${PREFIX}/${SPLIT}.json \
    --output ${SAVE_PATH} \
    --language ${LANGUAGE} \
    --framework ${PREFIX} \
    --clean_span $(python -c "print(int('${SPLIT}' not in ['dev', 'test']))")
  mtool --read mrp --analyze ${SAVE_PATH}
done

# multibooked_ca
PREFIX=multibooked_ca
LANGUAGE=ca
for SPLIT in train dev test
do
  SAVE_PATH=${SAVE_DIR}/${PREFIX}_${SPLIT}.${LANGUAGE}.mrp
  if [ -e ${SAVE_PATH} ]; then
    echo "${SAVE_PATH} already exists. Skipping."
    continue
  fi
  python -m utils.converter.ssa2mrp \
    --input ./data/orig/semeval22_structured_sentiment/data/${PREFIX}/${SPLIT}.json \
    --output ${SAVE_PATH} \
    --language ${LANGUAGE} \
    --framework ${PREFIX} \
    --clean_span $(python -c "print(int('${SPLIT}' not in ['dev', 'test']))")
  mtool --read mrp --analyze ${SAVE_PATH}
done

# multibooked_eu
PREFIX=multibooked_eu
LANGUAGE=eu
for SPLIT in train dev test
do
  SAVE_PATH=${SAVE_DIR}/${PREFIX}_${SPLIT}.${LANGUAGE}.mrp
  if [ -e ${SAVE_PATH} ]; then
    echo "${SAVE_PATH} already exists. Skipping."
    continue
  fi
  python -m utils.converter.ssa2mrp \
    --input ./data/orig/semeval22_structured_sentiment/data/${PREFIX}/${SPLIT}.json \
    --output ${SAVE_PATH} \
    --language ${LANGUAGE} \
    --framework ${PREFIX} \
    --clean_span $(python -c "print(int('${SPLIT}' not in ['dev', 'test']))")
  mtool --read mrp --analyze ${SAVE_PATH}
done

# opener_en
PREFIX=opener_en
LANGUAGE=en
for SPLIT in train dev test
do
  SAVE_PATH=${SAVE_DIR}/${PREFIX}_${SPLIT}.${LANGUAGE}.mrp
  if [ -e ${SAVE_PATH} ]; then
    echo "${SAVE_PATH} already exists. Skipping."
    continue
  fi
  python -m utils.converter.ssa2mrp \
    --input ./data/orig/semeval22_structured_sentiment/data/${PREFIX}/${SPLIT}.json \
    --output ${SAVE_PATH} \
    --language ${LANGUAGE} \
    --framework ${PREFIX} \
    --clean_span $(python -c "print(int('${SPLIT}' not in ['dev', 'test']))")
  mtool --read mrp --analyze ${SAVE_PATH}
done

# opener_es
PREFIX=opener_es
LANGUAGE=es
for SPLIT in train dev test
do
  SAVE_PATH=${SAVE_DIR}/${PREFIX}_${SPLIT}.${LANGUAGE}.mrp
  if [ -e ${SAVE_PATH} ]; then
    echo "${SAVE_PATH} already exists. Skipping."
    continue
  fi
  python -m utils.converter.ssa2mrp \
    --input ./data/orig/semeval22_structured_sentiment/data/${PREFIX}/${SPLIT}.json \
    --output ${SAVE_PATH} \
    --language ${LANGUAGE} \
    --framework ${PREFIX} \
    --clean_span $(python -c "print(int('${SPLIT}' not in ['dev', 'test']))")
  mtool --read mrp --analyze ${SAVE_PATH}
done

# darmstadt_unis
PREFIX=darmstadt_unis
LANGUAGE=en
for SPLIT in train dev test
do
  SAVE_PATH=${SAVE_DIR}/${PREFIX}_${SPLIT}.${LANGUAGE}.mrp
  if [ -e ${SAVE_PATH} ]; then
    echo "${SAVE_PATH} already exists. Skipping."
    continue
  fi
  python -m utils.converter.ssa2mrp \
    --input ./data/orig/semeval22_structured_sentiment/data/${PREFIX}/${SPLIT}.json \
    --output ${SAVE_PATH} \
    --language ${LANGUAGE} \
    --framework ${PREFIX} \
    --clean_span $(python -c "print(int('${SPLIT}' not in ['dev', 'test']))")
  mtool --read mrp --analyze ${SAVE_PATH}
done

# mpqa
PREFIX=mpqa
LANGUAGE=en
for SPLIT in train dev test
do
  SAVE_PATH=${SAVE_DIR}/${PREFIX}_${SPLIT}.${LANGUAGE}.mrp
  if [ -e ${SAVE_PATH} ]; then
    echo "${SAVE_PATH} already exists. Skipping."
    continue
  fi
  python -m utils.converter.ssa2mrp \
    --input ./data/orig/semeval22_structured_sentiment/data/${PREFIX}/${SPLIT}.json \
    --output ${SAVE_PATH} \
    --language ${LANGUAGE} \
    --framework ${PREFIX} \
    --clean_span $(python -c "print(int('${SPLIT}' not in ['dev', 'test']))")
  mtool --read mrp --analyze ${SAVE_PATH}
done
