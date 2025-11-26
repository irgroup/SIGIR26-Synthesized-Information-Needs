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



## 2 Generating Topics for Robust
||Data|Script|Status|
|---|---|---|---|
|Topics| [robust-topics](./data/interim/qrels-robust-topics-generated) | [topics-robust-topics.sh](./scripts/topics-robust-topics.sh) |DONE|
|Qrels| [qrels-robust-topics-generated](./data/interim/qrels-robust-topics-generated) | [qrels-robust-topics-generated.sh](./scripts/qrels-robust-topics-generated.sh) |DONE|

Prompts:
- [query](./data/raw/prompts/topic-query.yaml): Query(-variants) only 
- [docs-pos](./data/raw/prompts/topic-docs-pos.yaml): Relevant Documents
- [docs-neg](./data/raw/prompts/topic-docs-neg.yaml): Non-Relevant Documents
- [query-docs-pos](./data/raw/prompts/topic-query-docs-pos.yaml): Query(-variants) + Relevant Documents
- [query-docs-neg](./data/raw/prompts/topic-query-docs-neg.yaml): Query(-variants) + Non-Relevant Documents
- [contrastive](./data/raw/prompts/topic-contrastive.yaml): Relevant + Non-Relevant Documents
- [query-contrastive](./data/raw/prompts/topic-query-contrastive.yaml): Query(-variants) + Relevant + Non-Relevant Documents


### 2.1 **Alignment:** How well align generated qrels based on generated topics with the original qrels?
[Evaluation](./notebooks/evaluate-alignment.ipynb) 


### 2.2 **Clarity:** How is the agreement between different generated qrels using the generated topics?
[Evaluation](./notebooks/evaluate-clarity.ipynb)



### 2.3 **Distinguishability**: How well can the generated qrels based on generated topics differentiate between retrieval systems?
[Evaluation](./notebooks/evaluate-distinguishability.ipynb)


## 3 Generating Topics for TREC DL 2021 (Document)

Reference:
||Data|Script|Status|
|---|---|---|---|
|Topics| TREC | - |-|
|Qrels| [qrels-dl21-reference](./data/interim/qrels-dl21-reference/) |[qrels-dl21-reference.sh](./scripts/qrels-dl21-reference.sh)|WIP|

Experimental:

||Data|Script|Status|
|---|---|---|---|
|Topics| [topics-dl21](./data/interim/topics-dl21) | [topics-dl21.sh](./scripts/topics-dl21.sh) ||
|Qrels| [qrels-dl21-topics-generated](./data/interim/qrels-dl21-topics-generated) | [qrels-dl21-topics-generated.sh](./scripts/qrels-dl21-topics-generated.sh) ||
