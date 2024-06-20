# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

set -eu

# Install packages
pip install pip==20.2
pip install Cython==0.29.37
pip install numpy==1.19.0 --no-build-isolation

pip install -r examples/semeval2022task10/requirements.txt

# Required for seq2seq parser
pip install -r examples/semeval2022task10/seq2seq_parser/requirements.txt

# Make dataset and log directories
mkdir -p data/orig
mkdir -p log
