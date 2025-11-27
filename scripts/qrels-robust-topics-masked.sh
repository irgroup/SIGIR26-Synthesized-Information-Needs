# Generate qrels based on partial topics for Robust


MODEL=qwen3-30B-MT100-no-think
# MODEL=gpt-oss-120B-MT1000
DATASET=robust
CONNECTION=http://localhost:6543/v1

# Partial Topics
OUTPUT=../data/interim/qrels-robust-topics-masked
mkdir -p $OUTPUT

python ./gen-qrels.py \
    --model $MODEL \
    --data $DATASET \
    --connection $CONNECTION \
    --prompt ../data/raw/prompts/-DNA-zero-shot-masked-description-narrative.yaml \
    --output $OUTPUT 

python ./gen-qrels.py \
    --model $MODEL \
    --data $DATASET \
    --connection $CONNECTION \
    --prompt ../data/raw/prompts/-DNA-zero-shot-masked-description.yaml \
    --output $OUTPUT 

python ./gen-qrels.py \
    --model $MODEL \
    --data $DATASET \
    --connection $CONNECTION \
    --prompt ../data/raw/prompts/-DNA-zero-shot-masked-narrative.yaml \
    --output $OUTPUT 

python ./gen-qrels.py \
    --model $MODEL \
    --data $DATASET \
    --connection $CONNECTION \
    --prompt ../data/raw/prompts/-DNA-zero-shot-masked-title-description.yaml \
    --output $OUTPUT 

python ./gen-qrels.py \
    --model $MODEL \
    --data $DATASET \
    --connection $CONNECTION \
    --prompt ../data/raw/prompts/-DNA-zero-shot-masked-title-narrative.yaml \
    --output $OUTPUT 

python ./gen-qrels.py \
    --model $MODEL \
    --data $DATASET \
    --connection $CONNECTION \
    --prompt ../data/raw/prompts/-DNA-zero-shot-masked-title.yaml \
    --output $OUTPUT 

