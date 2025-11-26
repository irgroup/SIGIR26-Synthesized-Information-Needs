# Generate qrels based on the original robust topics
# DONE


# MODEL=qwen3-30B-MT100-no-think
# DATASET=robust
# CONNECTION=http://localhost:6542/v1


# # Original Topics
# PROMPT=-DNA-zero-shot
# OUTPUT=../data/interim/robust-reference
# python3 ./gen-qrels.py \
#     --model $MODEL \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt $PROMPT \
#     --output $OUTPUT 
