# conf26-Generating-Topics

## 1 Pre Requisites
[Evaluation](./notebooks/evaluate-pre-requisites.ipynb)

### 1.1 Can we reproduce the qrels quality by Thomas et al. with open models?
||Data|Script|Status|
|---|---|---|---|
|Topics| TREC | - |-|
|Qrels| [robust-reference](./data/interim/robust-reference/) |[qrels-robust-reference.sh](./scripts/qrels-robust-reference.sh)|DONE|


### 1.2 How do different topic components influence the qrel generation?
||Data|Script|Status|
|---|---|---|---|
|Topics| TREC |-|-|
|Qrels| [qrels-robust-topics-masked](./data/interim/qrels-robust-topics-masked/) |[qrels-robust-topics-masked.sh](./scripts/qrels-robust-topics-masked.sh)|DONE|



## Generating Topics for Robust
||Data|Script|Status|
|---|---|---|---|
|Topics| [robust-topics](./data/interim/qrels-robust-topics-generated) | [topics-robust-topics.sh](./scripts/topics-robust-topics.sh) |DONE|
|Qrels| [qrels-robust-topics-generated](./data/interim/qrels-robust-topics-generated) | [qrels-robust-topics-generated.sh](./scripts/qrels-robust-topics-generated.sh) |WIP|




#### Alignment
[Evaluation](./notebooks/evaluate-alignment.ipynb) 


## Clarity
[Evaluation](./notebooks/evaluate-clarity.ipynb)



## Distinguishability
[Evaluation](./notebooks/evaluate-distinguishability.ipynb)
