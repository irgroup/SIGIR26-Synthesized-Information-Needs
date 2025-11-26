#!/bin/bash
MODEL=qwen3-30B-MT100-no-think
DATASET=robust
CONNECTION=http://localhost:6542/v1

# Generated Topics
SEARCH_DIR="/workspaces/conf26-generating-topics/data/interim/robust-topics"
for dir in "$SEARCH_DIR"/*; do
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
