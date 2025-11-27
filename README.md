# conf26-Generating-Topics

## 1 Pre Requisites
[Evaluation](./notebooks/evaluate-pre-requisites.ipynb)

### 1.1 Can we reproduce the qrels quality by Thomas et al. with open models?
||Data|Script|Qwen3-30B|GPT-OSS-120B|
|---|---|---|---|---|
|Topics| TREC | - |-| -|
|Qrels| [qrels-robust-reference](./data/interim/qrels-robust-reference/) |[qrels-robust-reference.sh](./scripts/qrels-robust-reference.sh)| DONE | |


### 1.2 How do different topic components influence the qrel generation?
||Data|Script|Qwen3-30B|GPT-OSS-120B|
|---|---|---|---|---|
|Topics| TREC |-|-|-|
|Qrels| [qrels-robust-topics-masked](./data/interim/qrels-robust-topics-masked/) |[qrels-robust-topics-masked.sh](./scripts/qrels-robust-topics-masked.sh)| DONE |  |



## 2 Controled Setting: Robust Test Collection
||Data|Script|Qwen3-30B|GPT-OSS-120B|
|---|---|---|---|---|
|Topics| [topics-robust](./data/interim/topics-robust) | [topics-robust.sh](./scripts/topics-robust.sh) |DONE|  |
|Qrels| [qrels-robust-topics-generated](./data/interim/qrels-robust-topics-generated) | [qrels-robust-topics-generated.sh](./scripts/qrels-robust-topics-generated.sh) | DONE | |

Prompts:
- [query](./data/raw/prompts/topic-query.yaml): Query(-variants) only 
- [docs-pos](./data/raw/prompts/topic-docs-pos.yaml): Relevant Documents
- [docs-neg](./data/raw/prompts/topic-docs-neg.yaml): Non-Relevant Documents
- [query-docs-pos](./data/raw/prompts/topic-query-docs-pos.yaml): Query(-variants) + Relevant Documents
- [query-docs-neg](./data/raw/prompts/topic-query-docs-neg.yaml): Query(-variants) + Non-Relevant Documents
- [contrastive](./data/raw/prompts/topic-contrastive.yaml): Relevant + Non-Relevant Documents
- [query-contrastive](./data/raw/prompts/topic-query-contrastive.yaml): Query(-variants) + Relevant + Non-Relevant Documents

run all:
```bash
bash qrels-robust-reference.sh && bash qrels-robust-topics-masked.sh && bash topics-robust.sh && bash qrels-robust-topics-generated.sh
```

### 2.1 **Alignment:** How well align generated qrels based on generated topics with the original qrels?
[Evaluation](./notebooks/evaluate-alignment.ipynb) 


### 2.2 **Clarity:** How is the agreement between different generated qrels using the generated topics?
[Evaluation](./notebooks/evaluate-clarity.ipynb)



### 2.3 **Distinguishability**: How well can the generated qrels based on generated topics differentiate between retrieval systems?
[Evaluation](./notebooks/evaluate-distinguishability.ipynb)


## 3 Scenario Setting:
### *3.1 TREC DL 2021 (Document)*

**Reference:**
||Data|Script|Qwen3-30B|GPT-OSS-120B|
|---|---|---|---|---|
|Topics| TREC | - |-|-|
|Qrels| [qrels-dl21-reference](./data/interim/qrels-dl21-reference/) |[qrels-dl21-reference.sh](./scripts/qrels-dl21-reference.sh)|| |



**Experimental:**
||Data|Script|Qwen3-30B|GPT-OSS-120B|
|---|---|---|---|---|
|Topics| [topics-dl21](./data/interim/topics-dl21) | [topics-dl21.sh](./scripts/topics-dl21.sh) | | 
|Qrels| [qrels-dl21-topics-generated](./data/interim/qrels-dl21-topics-generated) | [qrels-dl21-topics-generated.sh](./scripts/qrels-dl21-topics-generated.sh) |||


### *3.2 LongEval-2023*
- ...


### ???

