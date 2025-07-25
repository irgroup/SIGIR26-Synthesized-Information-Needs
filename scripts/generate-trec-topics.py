"""Generate TREC topics

Example usage:
    python scripts/generate-trec-topics.py --dataset robust --llm openai --prompt uqv --nqueries 5 --ndocs 10 --output ./output
"""

import json
import logging
import os
from pathlib import Path

import click
from topic_gen import logger
from topic_gen.generate import Generator
from topic_gen.models import TRECTopic
from tqdm import tqdm

from src.data import get_dataset
from src.llm_connections import get_llm

logger.setLevel("INFO")


def add_logger(path_to_log):
    file_handler = logging.FileHandler(f"{path_to_log}-generation.log")
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # File handler


@click.command()
@click.option("--dataset", type=str, help="Dataset to use for topic generation")
@click.option("--llm", type=str, help="LLM to use for topic generation")
@click.option("--prompt", type=str, help="Prompt template to use for topic generation")
@click.option("--nqueries", type=int, default=5, help="Number of queries to use")
@click.option("--ndocs", type=int, default=10, help="Number of documents to use")
@click.option("--output", type=Path, help="Output directory for generated topics")
def main(dataset, llm, prompt, nqueries, ndocs, output):
    file_name = f"topics-{dataset}-{llm}-{prompt}-q{nqueries}-d{ndocs}"
    add_logger(os.path.join(output, file_name))

    llm_connection = get_llm(llm)
    topics = get_dataset(dataset)

    generator = Generator(llm=llm_connection, output_class=TRECTopic)
    print(f"Generating topics for {len(topics[200:210])} queries...")

    for topic in tqdm(topics[200:210]):
        try:
            generated_topic = generator.generate(
                prompt_name=prompt,
                number_of_topics=1,
                queries="\n".join(topic["uqv"][:nqueries]),
                relevant_documents="\n\n".join(topic["rel_docs"][:ndocs]),
            )

            # serialize
            topic_dict = generated_topic.model_dump()

            # add original id
            topic_dict["qid"] = topic["qid"]

            # write
            with open(os.path.join(output, file_name) + ".jsonl", "a+") as f:
                f.write(json.dumps(topic_dict) + "\n")

        except Exception as e:
            logger.error(f"Error generating topic for query {topic['qid']}: {e}")


if __name__ == "__main__":
    main()
