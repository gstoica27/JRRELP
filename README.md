# JRRELP

 ```$xslt
Knowledge Graph Enhanced Relation Extraction
George Stoica, Emmanouil Antonios Platanios, and Barnabás Póczos
NeurIPS 2020 KR2ML Workshop
```

This repository contains all code relevant to JRRELP, a general multitask learning framework for improving relation extraction via link prediction.

For details on this work please check out our:
* arXiv: [Paper](https://arxiv.org/abs/2012.04812)
* NeurIPS 2020 KR2ML Workshop: [Paper](https://kr2ml.github.io/2020/papers/KR2ML_28_paper.pdf) & [Poster](https://kr2ml.github.io/2020/papers/KR2ML_28_poster.pdf)

# Running Experiments
We base our code off of three open-source repositories: PA-LSTM, C-GCN, & SpanBERT. To run JRRELP in a model, navigate to its respective named directory. We will use PALSTM as a running example.

1) Index into model directory
2) Download & prepare data by following instructions in the observed README.md file.
2) Run experiment by: 
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
3) To change parameters of an experiment, all you need to do is modify either "base_config.yaml" or "kglp_config.yaml" in the configs directory. Each config file has parameter comments highlighting what each parameter does. 
