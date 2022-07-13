# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

# Make dataset and log directories
mkdir -p data
mkdir -p log

chmod 755 ./examples/multitask_am/validate_data.sh
chmod 755 ./examples/multitask_am/setup_gold_graphs.sh
chmod 755 ./examples/multitask_am/setup_preprocess_all.sh
chmod 755 ./examples/multitask_am/setup_mtc.sh
chmod 755 ./examples/multitask_am/setup_cdcp.sh
chmod 755 ./examples/multitask_am/setup_abstrct.sh
chmod 755 ./examples/multitask_am/setup_aasd.sh
chmod 755 ./examples/multitask_am/setup_aaec.sh
chmod 755 ./examples/multitask_am/run_test.sh

# Preprocess datasets
./examples/multitask_am/setup_preprocess_all.sh

# Run python unittest
./examples/multitask_am/run_test.sh
