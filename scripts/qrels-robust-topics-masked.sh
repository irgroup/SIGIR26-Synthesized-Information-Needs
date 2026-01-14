#!/bin/bash
# Generate qrels based on original and partial TREC topics for Robust 2004

# MODEL=qwen3-30B-no-think
# CONNECTION=http://localhost:6543/v1
# MAX_CONCURRENCY=6

MODEL=qwen3-80B-next-no-think
CONNECTION=http://139.6.160.34:6543/v1
MAX_CONCURRENCY=40

# MODEL=gpt-oss-20B
# CONNECTION=http://localhost:6544/v1
# MAX_CONCURRENCY=20

# MODEL=gpt-oss-120Bjueri

# CONNECTION=http://localhost:6543/v1
# MAX_CONCURRENCY=40

# MODEL=llama3-3-70b_instruct_q8
# CONNECTION=http://139.6.160.34:6543/v1
# MAX_CONCURRENCY=4

DATASET=robust
OUTPUT=../data/interim/robust/qrels-topics-masked

# Original Topics
PROMPT=-DNA-zero-shot
# python ./gen-qrels.py \
#     --model $MODEL \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt $PROMPT \
#     --max_concurrency $MAX_CONCURRENCY \
#     --output $OUTPUT


# # Partial Topics
# python ./gen-qrels.py \
#     --model $MODEL \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt ../data/raw/prompts/-DNA-zero-shot-masked-description-narrative.yaml \
#     --max_concurrency $MAX_CONCURRENCY \
#     --output $OUTPUT 

# python ./gen-qrels.py \
#     --model $MODEL \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt ../data/raw/prompts/-DNA-zero-shot-masked-description.yaml \
#     --max_concurrency $MAX_CONCURRENCY \
#     --output $OUTPUT 

# python ./gen-qrels.py \
#     --model $MODEL \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt ../data/raw/prompts/-DNA-zero-shot-masked-narrative.yaml \
#     --max_concurrency $MAX_CONCURRENCY \
#     --output $OUTPUT 

# python ./gen-qrels.py \
#     --model $MODEL \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt ../data/raw/prompts/-DNA-zero-shot-masked-title-description.yaml \
#     --max_concurrency $MAX_CONCURRENCY \
#     --output $OUTPUT 

# python ./gen-qrels.py \
#     --model $MODEL \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt ../data/raw/prompts/-DNA-zero-shot-masked-title-narrative.yaml \
#     --max_concurrency $MAX_CONCURRENCY \
#     --output $OUTPUT 

python ./gen-qrels.py \
    --model $MODEL \
    --data $DATASET \
    --connection $CONNECTION \
    --prompt ../data/raw/prompts/-DNA-zero-shot-masked-title.yaml \
    --max_concurrency $MAX_CONCURRENCY \
    --output $OUTPUT 
