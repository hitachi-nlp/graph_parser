#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


OUT_DIR=log/multitask_am/gold_graphs

python -m utils.visualize_graphs --input data/input/aaec_essay.mrp --log ${OUT_DIR} --formats pdf --cores 4
python -m utils.visualize_graphs --input data/input/aaec_para.mrp --log ${OUT_DIR} --formats pdf --cores 4
python -m utils.visualize_graphs --input data/input/cdcp.mrp --log ${OUT_DIR} --formats pdf --cores 4
python -m utils.visualize_graphs --input data/input/mtc.mrp --log ${OUT_DIR} --formats pdf --cores 4
python -m utils.visualize_graphs --input data/input/abstrct.mrp --log ${OUT_DIR} --formats pdf --cores 4
python -m utils.visualize_graphs --input data/input/aasd.mrp --log ${OUT_DIR} --formats pdf --cores 4
