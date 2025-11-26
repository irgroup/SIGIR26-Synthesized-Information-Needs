#!/bin/bash
MODEL=qwen3-30B-MT100-no-think
DATASET=robust
CONNECTION=http://localhost:6542/v1

# Generated Topics
TOPICS_DIR="/workspaces/conf26-generating-topics/data/interim/topics-dl21"
for dir in "$TOPICS_DIR"/*; do
    if [ -d "$dir" ]; then
        python ./gen-qrels.py \
            --model $MODEL \
            --data $DATASET \
            --connection $CONNECTION \
            --prompt -DNA-zero-shot \
            --topics "$dir" \
            --output /workspaces/conf26-generating-topics/data/interim/qrels-dl21-topics-generated
    fi
done
