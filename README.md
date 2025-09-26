# conf26-Generating-Topics


## Experiments:

### 1. Input size: How many queries and how many documents influence the topic similarity


### 2. Reference: How good are LLM judgments on this dataset?
Generate judgments for Robust
- [with the full topic](scripts/exp2-llm-judge.py)
- [with the title of the topic only](scripts/exp2.1-llm-judge.py)

[Analysis](notebooks/exp-2.ipynb)


### 3. Masked topic generation
- Generate topic **title** based on description and narrative
- Generate **description** based on title and narrative
- Generate **narrative** based on title and discription

- Generate qrels based on new topics


### 4. Generate qrels based on incomplete 
- Generate qrels based on **title** only
- Generate qrels based on **title**, **description** 
- Generate qrels based on **narrative** 
