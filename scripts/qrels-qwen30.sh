
MODEL=qwen3-30B-MT100-no-think
DATASET=robust
CONNECTION=http://localhost:6542/v1
OUTPUT=../data/processed/qrels
PROMPT=-DNA-zero-shot

# ## Original
# python ./gen-qrels.py \
#     --model $MODEL \
#     --s \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt $PROMPT \
#     --output $OUTPUT 

# ## Partial Topics Prompts
# python ./gen-qrels.py \
#     --model $MODEL \
#     --s \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt ../data/raw/prompts/-DNA-zero-shot-masked-description-narrative.yaml \
#     --output $OUTPUT 

# python ./gen-qrels.py \
#     --model $MODEL \
#     --s \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt ../data/raw/prompts/-DNA-zero-shot-masked-description.yaml \
#     --output $OUTPUT 

# python ./gen-qrels.py \
#     --model $MODEL \
#     --s \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt ../data/raw/prompts/-DNA-zero-shot-masked-narrative.yaml \
#     --output $OUTPUT 

# python ./gen-qrels.py \
#     --model $MODEL \
#     --s \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt ../data/raw/prompts/-DNA-zero-shot-masked-title-description.yaml \
#     --output $OUTPUT 

# python ./gen-qrels.py \
#     --model $MODEL \
#     --s \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt ../data/raw/prompts/-DNA-zero-shot-masked-title-narrative.yaml \
#     --output $OUTPUT 

# python ./gen-qrels.py \
#     --model $MODEL \
#     --s \
#     --data $DATASET \
#     --connection $CONNECTION \
#     --prompt ../data/raw/prompts/-DNA-zero-shot-masked-title.yaml \
#     --output $OUTPUT 

