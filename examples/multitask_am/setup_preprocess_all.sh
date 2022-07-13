#!/bin/bash
# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.


# Preprocess datasets
./examples/multitask_am/setup_aaec.sh
./examples/multitask_am/setup_cdcp.sh
./examples/multitask_am/setup_mtc.sh
./examples/multitask_am/setup_abstrct.sh
./examples/multitask_am/setup_aasd.sh

# Validate the preprocessed data
./examples/multitask_am/validate_data.sh

./examples/multitask_am/setup_gold_graphs.sh

