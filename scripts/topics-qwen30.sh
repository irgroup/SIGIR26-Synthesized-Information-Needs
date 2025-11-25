MODEL=qwen3-30B-no-think
DATASET=robust
CONNECTION=http://localhost:6542/v1
OUTPUT=../data/processed/topics

# # # TREC
# PROMPT=../data/raw/prompts/trec.yaml
# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 1 \
#     --ndocspos 1 

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 3 \
#     --ndocspos 2 

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 5 \
#     --ndocspos 3 

# # # Query
# PROMPT=../data/raw/prompts/trec-query.yaml
# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 1

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 3  

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 5 


# # Contrastive
# PROMPT=../data/raw/prompts/trec-contrastive.yaml
# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 1 \
#     --ndocspos 1 \
#     --ndocsneg 1

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 3 \
#     --ndocspos 2 \
#     --ndocsneg 2

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 5 \
#     --ndocspos 3 \
#     --ndocsneg 3

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 7 \
#     --ndocspos 4 \
#     --ndocsneg 4

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 10 \
#     --ndocspos 5 \
#     --ndocsneg 5


# # relevant documents
# PROMPT=../data/raw/prompts/trec-docs-rel.yaml
# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --ndocspos 1 

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --ndocspos 2 

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --ndocspos 3 

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --ndocspos 4 

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --ndocspos 5 



# # contrastive documents only
# PROMPT=../data/raw/prompts/trec-docs.yaml
# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --ndocspos 1 \
#     --ndocsneg 1

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --ndocspos 2 \
#     --ndocsneg 2

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --ndocspos 3 \
#     --ndocsneg 3

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --ndocspos 4 \
#     --ndocsneg 4

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --ndocspos 5 \
#     --ndocsneg 5

# Contrastiv No Variants
# PROMPT=../data/raw/prompts/trec-contrastive.yaml
# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 1 \
#     --ndocspos 3 \
#     --ndocsneg 3

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 1 \
#     --ndocspos 4 \
#     --ndocsneg 4

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 1 \
#     --ndocspos 5 \
#     --ndocsneg 5

# python ./gen-topics.py \
#     --model $MODEL \
#     --prompt  $PROMPT \
#     --connection $CONNECTION \
#     --output $OUTPUT \
#     --data $DATASET \
#     --nqueries 1 \
#     --ndocspos 6 \
#     --ndocsneg 6
