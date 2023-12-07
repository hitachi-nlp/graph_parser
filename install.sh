# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

set -eu

pip install pip==20.2

mkdir -p data/
mkdir -p log/

if [ ! -e amparse/common/chu_liu_edmonds.py ]; then
  wget https://raw.githubusercontent.com/allenai/allennlp/a558f7f4982d6a996d5397c7ff7a500cb0251577/allennlp/nn/chu_liu_edmonds.py
  mv chu_liu_edmonds.py amparse/common
  sed -i'' -e 's/from allennlp.common.checks import ConfigurationError//g' amparse/common/chu_liu_edmonds.py
fi

pip install -r requirements.txt

