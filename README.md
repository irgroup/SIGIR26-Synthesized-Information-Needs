# Formalized Information Needs Improve Large-Language-Model Relevance Judgments

[![Venue: SIGIR 2026](https://img.shields.io/badge/Venue-SIGIR%202026-blue.svg)](https://sigir2026.org)
[![arXiv](https://img.shields.io/badge/arXiv-2604.04140-b31b1b.svg)](https://arxiv.org/abs/2604.04140)
[![DOI: doi](https://img.shields.io/badge/DOI-10.1145/3805712.3809561-blue.svg)](https://doi.org/10.1145/3805712.3809561)


> **Formalized Information Needs Improve Large-Language-Model Relevance Judgments**
> Jüri Keller, Maik Fröbe, Björn Engelmann, Fabian Haak, Timo Breuer, Birger Larsen, and Philipp Schaer.
> *Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’26)*.

> [!TIP]
> The camera ready version is now available on [arXiv](https://arxiv.org/abs/2604.04140).

## Abstract
> Cranfield-style retrieval evaluations with too few or too many relevant documents or with low inter-assessor agreement on relevance can reduce the reliability of observations. In evaluations with human assessors, information needs are often formalized as retrieval topics to avoid an excessive number of relevant documents while maintaining good agreement. However, emerging evaluation setups that use Large Language Models (LLMs) as relevance assessors often use only queries, potentially decreasing the reliability. To study whether LLM relevance assessors benefit from formalized information needs, we synthetically formalize information needs with LLMs into topics that follow the established structure from previous human relevance assessments (i.e., descriptions and narratives). We compare assessors using synthetically formalized topics against the LLM-default query-only assessor on the 2019/2020 editions of TREC Deep Learning and Robust04. We find that assessors without formalization judge many more documents relevant and have a lower agreement, leading to reduced reliability in retrieval evaluations. Furthermore, we show that the formalized topics improve agreement between human and LLM relevance judgments, even when the topics are not highly similar to their human counterparts. Our findings indicate that LLM relevance assessors should use formalized information needs, as is standard for human assessment, and synthetically formalize topics when no human formalization exists to improve evaluation reliability.


## Setup
Install the dependencies listed in the [pyprojects.toml](./pyproject.toml) for example with poetry:
`uv sync`

## Results
All synthesized topics and LLM judgments are provided in the [data/interim](./data/interim/) directory.
Summarizing statistics such as topic and label similarity are provided in the [data/processed](./data/processed/) directory.


## Topic-Gen
The [Topic-Gen](./topic-gen/) toolkit is used to generate topics and qrels. 


## Synthesizing Topics
Topics are generated with the [gen-topics.py](./scripts/gen-topics.py) script.

### Prompts
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


## LLM Relevance Assessments
Qrels are generated with the [gen-qrels.py](./scripts/gen-qrels.py) script.

### Prompts
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


## Evaluation
All evaluation scripts are available from the [scripts](./scripts/) directory.
