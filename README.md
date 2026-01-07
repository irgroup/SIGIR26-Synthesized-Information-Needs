# conf26-Generating-Topics


## Setup and Reproduction
Install the dependencies listed in the [pyprojects.toml](./pyproject.toml) for example with poetry:
`poetry install`

## Prompts
Topic generation:
- [query](./data/raw/prompts/topic-query.yaml): Query(-variants) only 
- [docs-pos](./data/raw/prompts/topic-docs-pos.yaml): Relevant Documents
- [docs-neg](./data/raw/prompts/topic-docs-neg.yaml): Non-Relevant Documents
- [query-docs-pos](./data/raw/prompts/topic-query-docs-pos.yaml): Query(-variants) + Relevant Documents
- [query-docs-neg](./data/raw/prompts/topic-query-docs-neg.yaml): Query(-variants) + Non-Relevant Documents
- [contrastive](./data/raw/prompts/topic-contrastive.yaml): Relevant + Non-Relevant Documents
- [query-contrastive](./data/raw/prompts/topic-query-contrastive.yaml): Query(-variants) + Relevant + Non-Relevant Documents

Masked topic generation:
- [topic-masked-title](data/raw/prompts/topic-masked-title.yaml)
- [topic-masked-description](data/raw/prompts/topic-masked-description.yaml)
- [topic-masked-narrative](data/raw/prompts/topic-masked-narrative.yaml)
- [topic-masked-title-description](data/raw/prompts/topic-masked-title-description.yaml)
- [topic-masked-title-narrative](data/raw/prompts/topic-masked-title-narrative.yaml)
- [topic-masked-description-narrative](data/raw/prompts/topic-masked-description-narrative.yaml) 

Qrel generation:
- [-DNA-zero-shot](topic-gen/topic_gen/prompts/-DNA-zero-shot.yaml)
- [qrel_zeroshot_bing](topic-gen/topic_gen/prompts/qrel_zeroshot_bing.yaml)

Masked qrel generation:
- [-DNA-zero-shot-masked-title](data/raw/prompts/-DNA-zero-shot-masked-title.yaml)
- [-DNA-zero-shot-masked-description](data/raw/prompts/-DNA-zero-shot-masked-description.yaml) 
- [-DNA-zero-shot-masked-narrative](data/raw/prompts/-DNA-zero-shot-masked-narrative.yaml) 
- [-DNA-zero-shot-masked-title-description](data/raw/prompts/-DNA-zero-shot-masked-title-description.yaml) 
- [-DNA-zero-shot-masked-title-narrative](data/raw/prompts/-DNA-zero-shot-masked-title-narrative.yaml) 
- [-DNA-zero-shot-masked-description-narrative](data/raw/prompts/-DNA-zero-shot-masked-description-narrative.yaml) 

## Models
- Qwen3-30B
- Qwen3-80B-Next
- gpt-oss-20B
- gpt-oss-120B
- llama3.3-70B-instruct-q8_0

## Datasets

### 1. Robust
|Task|Topics|Data|Script|Qwen3-30B|Qwen3-80N-Next|gpt-oss-20B|gpt-oss-120B|gpt-oss-120B-ollama|llama3.3-70B-instruct-q8_0|
|---|---|---|---|---|---|---|---|---|---|
|Qrels| (Masked) TREC |[qrels](./data/interim/robust/qrels-reference) |[script](./scripts/qrels-robust-reference.sh)| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|Topics|| [topics](./data/interim/robust/topics) |[script](./scripts/topics-robust.sh)| ✅ | ✅  | ✅ | ✅  | ✅ | ✅ |
|Masked Topics| partial TREC| [topics](./data/interim/robust/topics-masked) |[script](./scripts/topics-robust-masked.sh)| ✅ | ✅ | ✅ | ✅ |  |  |
|Qrels| Gen | [qrels](./data/interim/robust/qrels-topics-generated) |[script](./scripts/qrels-robust-topics-generated.sh)| ✅ | ✅ | ✅ | ✅ |  |  |
|Qrels|Masked Gen|tbd|tbd| ✅ | ✅ | ✅ | ✅ |  |  |

### 2. TREC-DL 2019
|Task|Topics|Data|Script|Qwen3-30B|Qwen3-80N-Next|gpt-oss-20B|gpt-oss-120B|llama3.3-70B-instruct-q8_0|
|---|---|---|---|---|---|---|---|---|
|Qrels| TREC |[qrels](./data/interim/dl19/qrels-reference) |[script](./scripts/qrels-dl19-reference.sh)| ✅ | ✅ | ✅ | ✅  | ✅ |
|Topics|| [topics](./data/interim/dl19/topics) |[script](./scripts/topics-dl19.sh)| ✅ | ✅ | ✅ | ✅ |  | ✅ |
|Qrels-Title| Gen | [qrels](./data/interim/dl19/qrels-topics-generated-title) |[script](./scripts/qrels-dl19-topics-generated.sh)| ✅ | ✅ | ✅ | ✅ |    |
|Qrels|Gen|[qrels](./data/interim/dl19/qrels-topics-generated-full) |[script](./scripts/qrels-dl19-topics-generated.sh)| ✅ | ✅ | wip | wip |  |  


### 3. TREC-DL 2020
|Task|Topics|Data|Script|Qwen3-30B|Qwen3-80N-Next|gpt-oss-20B|gpt-oss-120B|llama3.3-70B-instruct-q8_0|
|---|---|---|---|---|---|---|---|---|
|Qrels|TREC|[qrels](./data/interim/dl20/qrels-reference)|[script](./scripts/qrels-dl20-reference.sh)| ✅ | ✅  | ✅ | ✅ |  ✅ |
|Topics||[topics](./data/interim/dl20/topics) |[script](./scripts/topics-dl20.sh)| ✅ | ✅ | ✅ | ✅ |   ✅ |
|Qrels-Title|Gen|[qrels](./data/interim/dl20/qrels-topics-generated-title) |[script](./scripts/qrels-dl20-topics-generated.sh)| ✅ | wip | | ✅ |  |  
|Qrels| Gen | [qrels](./data/interim/dl20/qrels-topics-generated-full) |[script](./scripts/qrels-dl20-topics-generated.sh)|  |  |  |  |  |


### 4. LongEval 2023

|Task|Topics|Data|Script|Qwen3-30B|Qwen3-80N-Next|gpt-oss-20B|gpt-oss-120B|gpt-oss-120B-ollama|llama3.3-70B-instruct-q8_0|
|---|---|---|---|---|---|---|---|---|---|
|Qrels|Click/Human|[qrels](./data/interim/longeval/qrels-reference)|[script](./scripts/qrels-longeval-reference.sh)|✅||✅|✅|  |  |
|Topics||[topics](./data/interim/longeval/topics)|[script](./scripts/topics-longeval.sh)|✅||✅|✅||✅|
|Qrels|Gen|[qrels](./data/interim/longeval/qrels-topics-generated)|[script](./scripts/qrels-longeval-topics-generated.sh)| |  |  |  |  |  |

