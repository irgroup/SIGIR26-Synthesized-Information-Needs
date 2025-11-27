# Generate topics based on a variety of prompts and contexts for robust

MODEL=qwen3-30B-no-think
# MODEL=gpt-oss-120B
DATASET=robust
CONNECTION=http://localhost:6543/v1

# topic-query.yaml: Query
PROMPT=../data/raw/prompts/topic-query.yaml
python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 1 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 3 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 5

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 7 


# topic-docs-pos.yaml: Relevant Documents
PROMPT=../data/raw/prompts/topic-docs-pos.yaml
python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocspos 1

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocspos 2

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocspos 3

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocspos 4 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocspos 5


# topic-docs-neg.yaml: Non-Relevant Documents
PROMPT=../data/raw/prompts/topic-docs-neg.yaml
python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocsneg 1

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocsneg 2

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocsneg 3

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocsneg 4 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocsneg 5

# topic-query-docs-pos.yaml: Query + Relevant Documents
PROMPT=../data/raw/prompts/topic-query-docs-pos.yaml
python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 1 \
    --ndocspos 1

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 2 \
    --ndocspos 2 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 3 \
    --ndocspos 3 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 4 \
    --ndocspos 4 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 5 \
    --ndocspos 5


# topic-query-docs-neg.yaml: Query + Non-Relevant Documents
PROMPT=../data/raw/prompts/topic-query-docs-neg.yaml
python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 1 \
    --ndocsneg 1

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 2 \
    --ndocsneg 2 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 3 \
    --ndocsneg 3 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 4 \
    --ndocsneg 4 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 5 \
    --ndocsneg 5


# topic-contrastive.yaml: Relevant + Non-Relevant Documents
PROMPT=../data/raw/prompts/topic-contrastive.yaml
python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocspos 1 \
    --ndocsneg 1

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocspos 2 \
    --ndocsneg 2 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocspos 3 \
    --ndocsneg 3 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocspos 4 \
    --ndocsneg 4 

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --ndocspos 5 \
    --ndocsneg 5

# topic-query-contrastive.yaml: Query + Relevant + Non-Relevant Documents
PROMPT=../data/raw/prompts/topic-query-contrastive.yaml
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
    --nqueries 2 \
    --ndocspos 2 \
    --ndocsneg 2

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 3 \
    --ndocspos 3 \
    --ndocsneg 3

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 4 \
    --ndocspos 4 \
    --ndocsneg 4

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 5 \
    --ndocspos 5 \
    --ndocsneg 5

python ./gen-topics.py \
    --model $MODEL \
    --prompt  $PROMPT \
    --connection $CONNECTION \
    --output $OUTPUT \
    --data $DATASET \
    --nqueries 6 \
    --ndocspos 6 \
    --ndocsneg 6

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

