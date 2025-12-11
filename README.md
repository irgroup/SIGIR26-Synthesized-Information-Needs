# conf26-Generating-Topics


## Setup and Reproduction
Install the dependencies listed in the [pyprojects.toml](./pyproject.toml) for example with poetry:
`poetry install`



## 1 Pre Requisites
[Evaluation](./notebooks/evaluate-pre-requisites.ipynb)

### 1.1 Can we reproduce the qrels quality by Thomas et al. with open models?
||Data|Script|Qwen3-30B|gpt-oss-120B|gpt-oss-120B-ollama|gpt-oss-20B|llama3.1-8B-instruct|qwen3-14B|llama3.1-70B-instruct-q8_0|deepseek-V3.2|
|---|---|---|---|---|---|---|---|---|---|---|
|Topics| TREC | - | - | - | - | - | - | - | - | - |
|Qrels| [qrels-robust-reference](./data/interim/qrels-robust-reference/) |[qrels-robust-reference.sh](./scripts/qrels-robust-reference.sh)| DONE | DONE |DONE | DONE |DONE | DONE | DONE |DONE|


### 1.2 How do different topic components influence the qrel generation?
||Data|Script|Qwen3-30B|gpt-oss-120B|gpt-oss-120B-ollama|gpt-oss-20B|llama3.1-8B-instruct|qwen3-14B|llama3.1-70B-instruct-q8_0|deepseek-V3.2|
|---|---|---|---|---|---|---|---|---|---|---|
|Topics| TREC | - | - | - | - | - | - | - | - | - |
|Qrels| [qrels-robust-topics-masked](./data/interim/qrels-robust-topics-masked/) |[qrels-robust-topics-masked.sh](./scripts/qrels-robust-topics-masked.sh)| DONE |  | DONE | DONE |  |  | STOPPED ||



## 2 Controled Setting: Robust Test Collection
||Data|Script|Qwen3-30B|gpt-oss-120B|gpt-oss-120B-ollama|gpt-oss-20B|llama3.1-8B-instruct|qwen3-14B|llama3.1-70B-instruct-q8_0|deepseek-V3.2|
|---|---|---|---|---|---|---|---|---|---|---|
|Topics| [topics-robust](./data/interim/topics-robust) | [topics-robust.sh](./scripts/topics-robust.sh) |DONE| DONE | DONE | DONE | -- | DONE | DONE | -- | 
|Qrels| [qrels-robust-topics-generated](./data/interim/qrels-robust-topics-generated) | [qrels-robust-topics-generated.sh](./scripts/qrels-robust-topics-generated.sh) | DONE |  | WIP | WIP |  |   |   |  |  

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


### 3.1 LongEval-2023
Between the human annotations and the click model qrels, some qrels overlap. For the 45 queries set 99 and for the 31 qrels set 134. I checked the alignment between both with Cohen's kappa and found that it is really low (0.07 and 0.06).

### 3.1.2 How do generated qrels aligne with click model and human label?



### 3.2 TREC DL 2019

### 3.2.1 Can we reproduce the qrels quality by Upadhyay et al. with open models?
||Data|Script|Qwen3-30B|gpt-oss-120B|gpt-oss-120B-ollama|gpt-oss-20B|llama3.1-8B-instruct|qwen3-14B|llama3.1-70B-instruct-q8_0|deepseek-V3.2|
|---|---|---|---|---|---|---|---|---|---|---|
|Topics| TREC | - | - | - | - | - | - | - | - | - |
|Qrels| [qrels-dl19-reference](./data/interim/dl19/qrels-dl19-reference) |[qrels-dl19-reference.sh](./scripts/qrels-dl19-reference.sh)| DONE |  | DONE | DONE | | DONE | |DONE|

### 3.2.2 Alignment
||Data|Script|Qwen3-30B|gpt-oss-120B|gpt-oss-120B-ollama|gpt-oss-20B|llama3.1-8B-instruct|qwen3-14B|llama3.1-70B-instruct-q8_0|deepseek-V3.2|
|---|---|---|---|---|---|---|---|---|---|---|
|Topics| [topics-dl19](./data/interim/dl19/topics-dl19) |[topics-dl19.sh](./scripts/topics-dl19.sh) | DONE |  | DONE | DONE |  |  DONE | DONE |  |
|Qrels| [qrels-dl19-topics-generated](./data/interim/dl19/qrels-dl19-reference) |[qrels-dl19-topics-generated.sh](./scripts/qrels-dl19-reference.sh)| DONE |  |  | DONE | |  |  ||

