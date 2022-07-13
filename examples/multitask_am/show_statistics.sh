#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


echo "------- AAEC essay-level -------: "
mtool --read mrp --analyze data/input/aaec_essay.mrp
python -m utils.show_label_stats --input data/input/aaec_essay.mrp
echo ""
echo "------- AAEC paragraph-level -------: "
mtool --read mrp --analyze data/input/aaec_para.mrp
python -m utils.show_label_stats --input data/input/aaec_para.mrp
echo ""
echo "------- MTC -------: "
mtool --read mrp --analyze data/input/mtc.mrp
python -m utils.show_label_stats --input data/input/mtc.mrp
echo ""
echo "------- CDCP -------: "
mtool --read mrp --analyze data/input/cdcp.mrp
python -m utils.show_label_stats --input data/input/cdcp.mrp
echo ""
echo "------- AbstRCT -------: "
mtool --read mrp --analyze data/input/abstrct.mrp
python -m utils.show_label_stats --input data/input/abstrct.mrp
echo ""
echo "------- AASD -------: "
mtool --read mrp --analyze data/input/aasd.mrp
python -m utils.show_label_stats --input data/input/aasd.mrp

