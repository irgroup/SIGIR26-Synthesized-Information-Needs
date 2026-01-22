# Generating Retrieval Topics
Toolkit to generate retrieval topics from various contexts.


## Usage
Also see the example notebook for generating [TREC topics](https://github.com/irgroup/topic-gen/blob/main/examples/generate-topics-trec.ipynb), [author topics](https://github.com/irgroup/topic-gen/blob/main/examples/generate-topics-author.ipynb), and [qrels](https://github.com/irgroup/topic-gen/blob/main/examples/generate-qrels.ipynb).

### Generate a Topic
```Python
from topic_gen import logger
from topic_gen.generate import Generator
from topic_gen.models import iSearchTopic, Topics

# Setup langchain LLM connection
llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0,
)

# Create a prompt generator with the topic specifications and the LLM connection.
generator = Generator(llm=llm, output_class=Topics[iSearchTopic], prompt_name="isearch-base")

# Generate topics based on a prompt and various inputs
topics = generator.generate_one(
    dry_run=True,                               # Set to True to avoid actual API calls
    name="Author Name",                         # Name of the person for whom topics are generated 
    publications="../data/name/publications",   # Path to the publications data
    number_of_topics=5,                         # Number of topics to generate
)
```
