# SIGIR26-Formalized-Information-Needs

## Setup and Reproduction
Install the dependencies listed in the [pyprojects.toml](./pyproject.toml) for example with poetry:
`uv sync`

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
