#!/bin/bash
# Generate qrels based on generated topics for Robust 2004

# MODEL=qwen3-30B-no-think
# CONNECTION=http://localhost:6543/v1
# MAX_CONCURRENCY=6

MODEL=qwen3-80B-next-no-think
CONNECTION=http://139.6.160.34:6543/v1
MAX_CONCURRENCY=40

# MODEL=gpt-oss-20B
# CONNECTION=http://localhost:6544/v1
# MAX_CONCURRENCY=20

# MODEL=gpt-oss-120B
# CONNECTION=http://localhost:6543/v1
# MAX_CONCURRENCY=40

# MODEL=llama3-3-70b_instruct_q8
# CONNECTION=http://139.6.160.34:6543/v1
# MAX_CONCURRENCY=4


DATASET=robust

# Generated Topics
TOPICS_DIR="/workspaces/conf26-generating-topics/data/interim/robust/topics/GPT-OSS-20B"
for dir in "$TOPICS_DIR"/*; do
    if [ -d "$dir" ]; then
        python ./gen-qrels.py \
            --model $MODEL \
            --max_concurrency $MAX_CONCURRENCY \
            --data $DATASET \
            --connection $CONNECTION \
            --prompt -DNA-zero-shot \
            --topics "$dir" \
            --output /workspaces/conf26-generating-topics/data/interim/robust/qrels-topics-generated
    fi
done
