# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

set -eu

./install.sh

# Install packages
pip install pip==20.2
pip install torch==1.9.0 transformers==4.15.0 protobuf==3.16.0
pip install lxml tqdm stanza==1.1.1 dl-translate==0.2.5 pyconll==3.1.0

# Required for seq2seq parser
pip install -r examples/semeval2022task10/seq2seq_parser/requirements.txt

# Make dataset and log directories
mkdir -p data/orig
mkdir -p log
