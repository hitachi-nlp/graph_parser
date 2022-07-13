
This example shows the reproducing procedure for our SemEval paper: ```Hitachi at SemEval-2022 Task 10: Comparing Graph- and Seq2Seq-based Models Highlights Difficulty in Structured Sentiment Analysis```.
The graph-based models achieved the third place.

## Environment

- Required: Python 3.8
- Recommended: Pyenv and Virtualenv

We tested the code on Linux and OSX.

To ensure reproducibility, please use following previous version:
```bash
git checkout example/semeval2022task10
```

## Install packages

Run the following commands under the project root directory:

```bash
# Install packages
./examples/semeval2022task10/install.sh


# Download stanza English resource
python -c "import stanza; stanza.download('en')"

# Download SemEval2022 Task 10 tools and data
git clone https://github.com/jerbarnes/semeval22_structured_sentiment.git
mkdir -p data/orig
mv semeval22_structured_sentiment data/orig
```

## Download MPQA and DSu

- MPQA

Go to the [MPQA 2.0](http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0/) website, agree to the license and download the corpus. Put the zipped archive in ```data/orig/semeval22_structured_sentiment/data/mpqa```.
Finally, run the extraction script.


```bash
cd data/orig/semeval22_structured_sentiment/data/mpqa
bash process_mpqa.sh
cd ../../../../../
```

- DSu

Go to the [Darmstadt Service Review Corpus website](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2448), agree to the license and download the corpus. Put the zipped archive in ```data/orig/semeval22_structured_sentiment/data/darmstadt_unis``` and finally, run the extraction script.


```bash
cd data/orig/semeval22_structured_sentiment/data/darmstadt_unis
bash process_darmstadt.sh
# For Mac OSX:
#bash process_darmstadt_OSX.sh
cd ../../../../../
```


## Graph-based method

### Data pre-processing

```bash
# Convert to the MRP format
./examples/semeval2022task10/convert_to_mrp.sh
# [Optional] Show graphs
./examples/semeval2022task10/show_gold_graphs.sh
```

### Training and evaluation

```bash
# Monolingual
./examples/semeval2022task10/graph_parser/monolingual/en_darmstadt_unis.sh
./examples/semeval2022task10/graph_parser/monolingual/en_mpqa.sh
./examples/semeval2022task10/graph_parser/monolingual/en_opener_en.sh
./examples/semeval2022task10/graph_parser/monolingual/ca.sh
./examples/semeval2022task10/graph_parser/monolingual/eu.sh
./examples/semeval2022task10/graph_parser/monolingual/es.sh
./examples/semeval2022task10/graph_parser/monolingual/no.sh

# Crosslingual-zeroshot
./examples/semeval2022task10/graph_parser/crosslingual_zeroshot/ca_es_eu.sh
```

Example of showing test data scores:
```bash
cat log/semeval2022task10/graph_monolingual/en_opener_en/[0-2]/test/eval.txt
```

Example of showing predicted test data:
```bash
cat log/semeval2022task10/graph_monolingual/en_opener_en/0/test/predictions.json | jq .
```


## Seq2seq method

### Data pre-processing

```bash
# Convert to the serialized format
./examples/semeval2022task10/seq2seq_parser/scripts/data_conversion.sh
```


### Training and evaluation

To ensure reproducibility, use 2 x Tesla V100 GPUs.

```bash
# Monolingual
./examples/semeval2022task10/seq2seq_parser/scripts/en_darmstadt_unis.sh
./examples/semeval2022task10/seq2seq_parser/scripts/en_mpqa.sh
./examples/semeval2022task10/seq2seq_parser/scripts/en_opener_en.sh
./examples/semeval2022task10/seq2seq_parser/scripts/ca.sh
./examples/semeval2022task10/seq2seq_parser/scripts/eu.sh
./examples/semeval2022task10/seq2seq_parser/scripts/es.sh
./examples/semeval2022task10/seq2seq_parser/scripts/no.sh

# Crosslingual-zeroshot
./examples/semeval2022task10/seq2seq_parser/scripts/crosslingual_zeroshot/ca_es_eu.sh
```

Example of showing test data scores:
```bash
cat log/semeval2022task10/seq2seq_monolingual/mpqa-t5-large-[0-2]/test/eval.txt
```

Example of showing predicted test data:
```bash
cat log/semeval2022task10/seq2seq_monolingual/mpqa-t5-large-0/test/predictions.json | jq .
```

## Citation

```
@inproceedings{morio_etal_2022_semeval,
    author = {Morio, Gaku and Ozaki, Hiroaki and Yamaguchi, Atsuki and Sogawa, Yasuhiro},
    title = "{Hitachi at SemEval-2022 Task 10: Comparing Graph- and Seq2Seq-based Models Highlights Difficulty in Structured Sentiment Analysis}",
    booktitle = "Proceedings of the Sixteenth Workshop on Semantic Evaluation",
    month = jul,
    year = "2022",
    address = "Seattle",
    publisher = "The Association for Computational Linguistics",
    pages = "to appear",
}
```
