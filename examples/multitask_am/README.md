This example shows the reproducing procedure for our TACL paper: ```End-to-end Argument Mining with Cross-corpora Multi-task Learning```.


## Environment

- Required: Python 3.8
- Recommended: Pyenv and Virtualenv
  
We tested the code on Linux and OSX.

To ensure reproducibility, please use following previous version:
```bash
git checkout example/mtam
```


## Install packages

```bash
./examples/multitask_am/install.sh
```

## Download and pre-process corpora

Before running the pre-processing code, you must see license information:
- AAEC: https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422
- AbstRCT: https://gitlab.com/tomaye/abstrct/
- MTC: https://github.com/peldszus/arg-microtexts

```bash
# Download and pre-process corpora
./examples/multitask_am/setup_all.sh
# [Optional] Show corpora statistics
./examples/multitask_am/show_statistics.sh
```

## Experiments

### Single-task (i.e., single corpus, ST) training

Training and evaluation:
```bash
# AAEC (UKP corpus in Essay-level)
./examples/multitask_am/st/aaec_essay.sh
# AAEC (UKP corpus in Paragraph-level)
./examples/multitask_am/st/aaec_para.sh
# AASD (SciDTB with argument annotation)
./examples/multitask_am/st/aasd.sh
# AbstRCT (Abstracts from RCT domain)
./examples/multitask_am/st/abstrct.sh
# CDCP (User comments for Consumer Debt Collection Practices)
./examples/multitask_am/st/cdcp.sh
# MTC (Microtext corpus)
./examples/multitask_am/st/mtc.sh
```

Example of showing test data scores:
```bash
# Span identification
cat log/multitask_am/st/aaec_para/*/finetune/evaluation/test.jsonl | jq .scores.aaec_para.anchors.f
# Component classification
cat log/multitask_am/st/aaec_para/*/finetune/evaluation/test.jsonl | jq .scores.aaec_para.labels.total.f
# Relation classification
cat log/multitask_am/st/aaec_para/*/finetune/evaluation/test.jsonl | jq .scores.aaec_para.edges.total.f
# Link identification
cat log/multitask_am/st/aaec_para/*/finetune/evaluation/test.jsonl | jq .scores.aaec_para.edges.link.f


# [w/ oracle span] Component classification
cat log/multitask_am/st/aaec_para/*/finetune/os/evaluation/test.jsonl | jq .labels.total.f
# [w/ oracle span] Relation classification
cat log/multitask_am/st/aaec_para/*/finetune/os/evaluation/test.jsonl | jq .edges.total.f
# [w/ oracle span] Link identification
cat log/multitask_am/st/aaec_para/*/finetune/os/evaluation/test.jsonl | jq .edges.link.f
```

Example of showing predicted test data:
```bash
# Only shows the first sample
cat log/multitask_am/st/aaec_para/0/finetune/prediction/test.mrp | head -1 | jq .
# [w/ oracle span]
cat log/multitask_am/st/aaec_para/0/finetune/os/prediction/test.mrp | head -1 | jq .
```

### Multi-task cross-corpora (i.e., multi-task using all the corpora, MT-All) training

Training and evaluation:
```bash
./examples/multitask_am/mt_all/aaec_essay.sh
./examples/multitask_am/mt_all/aasd.sh
./examples/multitask_am/mt_all/abstrct.sh
./examples/multitask_am/mt_all/cdcp.sh
./examples/multitask_am/mt_all/mtc.sh
```

The evaluation and prediction results can be found in the similar manner to the ST example.

## QAs

- Q: I can not reproduce exactly the same values as the paper
    - It is possible that you can not reproduce the exact values due to GPU environmental differences and so on (even using the same seed). 
      Also, Longformer seems to behave in a non-deterministic manner: see issues/12482 in the Huggingface transformers library.

- Q: Some corpus statistics are different from original papers
    - CDCP: Our pre-processing seems to a bit differ from other previous studies in how we handle link-transitivity. 
      However, we have the same statistics for the test data, so we believe there is no problem in comparing performance.
    - AbstRCT: The original statistics contained an erroneously double-annotated relationship. Our statistics corrected this.
    - AASD: The published corpus appears to contain minor modifications.
    - MTC: The pre-processing method for Add relations follows previous work. See comments in the pre-processing code.
    

## Citation

```
@article{10.1162/tacl_a_00481,
    author = {Morio, Gaku and Ozaki, Hiroaki and Morishita, Terufumi and Yanai, Kohsuke},
    title = "{End-to-end Argument Mining with Cross-corpora Multi-task Learning}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {10},
    pages = {639-658},
    year = {2022},
    month = {05},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00481},
    url = {https://doi.org/10.1162/tacl\_a\_00481},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00481/2022965/tacl\_a\_00481.pdf},
}
```

