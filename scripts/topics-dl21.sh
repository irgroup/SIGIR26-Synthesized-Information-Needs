# Generate topics based on a variety of prompts and contexts for robust

MODEL=qwen3-30B-no-think
DATASET=robust
CONNECTION=http://localhost:6542/v1
OUTPUT=../data/interim/topics-dl21


# topic-query.yaml: Query
PROMPT=../data/raw/prompts/topic-query.yaml
python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 1 


# without query variants
python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 1 \
    --ndocspos 1 \
    --ndocsneg 1

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 1 \
    --ndocspos 2 \
    --ndocsneg 2

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 1 \
    --ndocspos 3 \
    --ndocsneg 3

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 1 \
    --ndocspos 4 \
    --ndocsneg 4

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 1 \
    --ndocspos 5 \
    --ndocsneg 5

