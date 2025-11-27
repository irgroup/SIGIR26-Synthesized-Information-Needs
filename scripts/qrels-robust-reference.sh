# Generate qrels based on the original robust topics

MODEL=qwen3-30B-MT100-no-think
# MODEL=gpt-oss-120B-MT1000
DATASET=robust
CONNECTION=http://localhost:6543/v1


# Original Topics
PROMPT=-DNA-zero-shot
OUTPUT=../data/interim/qrels-robust-reference
python ./gen-qrels.py \
    --model $MODEL \
    --data $DATASET \
    --connection $CONNECTION \
    --prompt $PROMPT \
    --output $OUTPUT 
