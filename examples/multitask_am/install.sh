# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

set -eu

if [ ! -e amparse/common/chu_liu_edmonds.py ]; then
  wget https://raw.githubusercontent.com/allenai/allennlp/a558f7f4982d6a996d5397c7ff7a500cb0251577/allennlp/nn/chu_liu_edmonds.py
  mv chu_liu_edmonds.py amparse/common
  sed -i'' -e 's/from allennlp.common.checks import ConfigurationError//g' amparse/common/chu_liu_edmonds.py
fi

if [ ! -e utils/converter/argmicro/emnlp2015 ]; then
  git clone https://github.com/kuribayashi4/span_based_argumentation_parser.git
  mkdir -p utils/converter/argmicro/
  mv span_based_argumentation_parser/src/preprocess/emnlp2015 utils/converter/argmicro/
  rm -rf span_based_argumentation_parser
fi

# Install packages
pip install pip==20.2
pip install Cython==0.29.37
pip install numpy==1.19.0 --no-build-isolation

pip install -r examples/multitask_am/requirements.txt

