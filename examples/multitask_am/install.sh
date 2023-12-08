# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

set -eu

if [ ! -e utils/converter/argmicro/emnlp2015 ]; then
  git clone https://github.com/kuribayashi4/span_based_argumentation_parser.git
  mkdir -p utils/converter/argmicro/
  mv span_based_argumentation_parser/src/preprocess/emnlp2015 utils/converter/argmicro/
  rm -rf span_based_argumentation_parser
fi

# Install packages
pip install pip==20.2
pip install -r examples/multitask_am/requirements.txt

