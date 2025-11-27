#!/bin/bash
MODEL=qwen3-30B-MT100-no-think
# MODEL=gpt-oss-120B-MT1000
DATASET=robust
CONNECTION=http://localhost:6543/v1

# Generated Topics
TOPICS_DIR="/workspaces/conf26-generating-topics/data/interim/topics-robust"
for dir in "$TOPICS_DIR"/*; do
    if [ -d "$dir" ]; then
        python ./gen-qrels.py \
            --model $MODEL \
            --data $DATASET \
            --connection $CONNECTION \
            --prompt -DNA-zero-shot \
            --topics "$dir" \
            --output /workspaces/conf26-generating-topics/data/interim/qrels-robust-topics-generated
    fi
done
