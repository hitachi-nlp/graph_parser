## General usage


#### Installation

We recommend using Python 3.8+ as well as Pyenv and Virtualenv.

```
./install.sh
```

### 1. Data format (.mrp)

The data format of training, validation and test files should follow the [MRP format](http://mrp.nlpl.eu/2020/index.php?page=14#:~:text=Uniform%20Graph%20Interchange%20Format) (<b>jsonline</b>) which was introduced in the CoNLL 2019 and 2020 shared tasks.
Different from the original MRP format, one should not contain ```properties``` and should not use overlap anchors.

Example ```train.mrp```:
``` json
{"id": "001", "input": "This is the sentence A. I show the sentence B. Finally, you see the sentence C.", "framework": "sample_tree_corpus", "time": "2020-08-05", "flavor": 0, "language": "en", "version": 1.0, "provenance": "temp", "source": "tmp", "nodes": [{"id": 0, "label": "Claim", "anchors": [{"from": 0, "to": 22}]}, {"id": 1, "label": "Premise", "anchors": [{"from": 24, "to": 45}]}, {"id": 2, "label": "Premise", "anchors": [{"from": 56, "to": 78}]}], "edges": [{"source": 0, "target": 1, "label": "Support"}, {"source": 0, "target": 2, "label": "Attack"}], "tops": [0]}
{"id": "002", ...}
...
```
See [some examples](DATA_FORMAT.md) for more format detail.



### 2. Training

``` bash
python -m amparse.trainer.train \
    --log ./log/train \
    --ftrain train.mrp \
    --seed 42 \
    --model_name_or_path "allenai/longformer-base-4096" \
    --build_numericalizer_on_entire_corpus true \
    --batch_size 4 \
    --eval_batch_size 16 \
    --postprocessor "sample_tree_corpus:tree,sample_trees_corpus:trees,sample_graph_corpus:graph" \
    --embed_dropout 0.1 \
    --mlp_dropout 0.1 \
    --dim_mlp 512 \
    --dim_biaffine 512 \
    --lambda_bio 1. \  # if you want to train only relations, set to 0.0
    --lambda_proposition 0.1 \  # if you want to train only relations, set to 0.0
    --lambda_arc 1. \  # if you want to train only components, set to 0.0
    --lambda_rel 0.1 \  # if you want to train only components, set to 0.0
    --lr 5e-5 \
    --beta1 0.9 \
    --beta2 0.998 \
    --warmup_ratio 0.1 \
    --clip 1.0 \
    --epochs 20 \
    --terminate_epochs 20 \
    --evaluate_epochs 20 \
    --evaluate_with_oracle_span false  # if you want to train only relations, set to true

# -> The trained model will be saved at ./log/train/model
```

Some key options (for more detail, use ```python -m amparse.trainer.train --help```):

| Name | Description |
| --- | --- |
| log  | The output directory path |
| ftrain | Train file (*.mrp) |
| fvalid | [Optional] Dev file (*.mrp) | 
| ftest | [Optional] Test file (*.mrp) |
| model_name_or_path | (i) Downloads the pre-trained model when Huggingface model name is specified. Otherwise, (ii) uses a trained model on the local path. |
| postprocessor | Specify the post-processor script for each framework. In default, we provide following post-processors:<br>```tree```: The graph is composed of a tree.<br>```trees```: The graph is composed of a set of trees.<br>```graph```: The graph does not form a tree.<br>See [some examples](misc/DATA_FORMAT.md) for more format detail. |




### 3. Prediction

``` bash
python -m amparse.predictor.predict \
      --log ./log/predict \
      --models ./log/train/model \
      --input test.mrp \
      --batch_size 16 \
      --oracle_span false

# -> The predicted result (an MRP file) will be saved at ./log/predict/prediction.mrp
```

Some key options (for more detail, use ```python -m amparse.predictor.predict --help```):

| Name | Description |
| --- | --- |
| log  | The output directory path |
| models | The trained model path. When multiple models are specified, the ensemble prediction will be enabled. |
| input | Input file (*.mrp) | 
| oracle_span | Whether to use oracle span (i.e., gold node anchors) when prediction |


### 4. Evaluation

```bash
python amparse/evaluator/scorer.py \
  -gold gold.mrp \
  -system prediction.mrp
```

The results contain following values:

| Key | Description |
| --- | --- |
| g | The number of gold positive samples |
| s | The number of predicted positive samples |
| c | The number of correctly predicted positive samples | 
| p | Precision (= c / s) |
| r | Recall (= c / g) |
| f | F1 (= (2 * p * r) / (p + r)) |

