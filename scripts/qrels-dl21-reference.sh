# Generate qrels based on the original robust topics

MODEL=qwen3-30B-MT100-no-think
DATASET=trec-dl-2021
CONNECTION=http://localhost:6542/v1


# Original Topics
PROMPT=-DNA-zero-shot-dl
OUTPUT=../data/interim/qrels-dl21-reference
python3 ./gen-qrels.py \
    --model $MODEL \
    --data $DATASET \
    --connection $CONNECTION \
    --prompt $PROMPT \
    --output $OUTPUT 
