#!/usr/bin/env python3
"""
Generate TREC Topics

Example usage:
    python scripts/gen-topics.py --model gpt-oss-120b --k 1 --prompt trec-base

"""
import click
import time
import os
from pathlib import Path

from topic_gen.generate import Generator
from topic_gen.models import Topics
from topic_gen import logger
from src.data import get_dataset, make_file_name, alter_class
from src.config import get_llm

logger.setLevel("DEBUG")


@click.command()
# LLM
@click.option("--model", help="The model to use. This must be registered in src/config.py", required=True, type=str)
@click.option("--max_concurrency", help="Maximum number of concurrent requests.", default=50, type=int)
@click.option("--connection", help="The connection string for the LLM.", default="http://localhost:6544/v1", type=str)
@click.option("--gpus", help="CUDA_VISIBLE_DEVICES", default="0", type=str)
# Generation options
@click.option("--data", help="Test collection to be loaded", type=str, required=True)
@click.option("--k", help="Generate results only for the first k samples.", default=None, type=int)
@click.option("--prompt", help="Prompt file path", type=click.Path(), required=True)
@click.option("--nqueries", help="Number of query variants.", default=5, type=int)
@click.option("--ndocs", help="Number of relevant documents.", default=3, type=int)
@click.option("--output", help="Output file path", type=click.Path(), default=".")
def main(model, max_concurrency, connection, gpus, data, k, prompt, nqueries, ndocs, output):
    # Get LLM
    llm = get_llm(model, connection=connection, gpus=gpus)

    # Load data
    dataset = get_dataset(dataset_name=data)

    components = dataset.topic_components(
        k=k, sample_queries=nqueries, sample_docs=ndocs)

    # determine output class
    output_class = alter_class(prompt, dataset.topic_class)

    # Setup generator
    config = {}
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    generator = Generator(llm=llm, output_class=output_class,
                          prompt=prompt, config=config)

    # Generate topics
    start_time = time.time()

    generated_topics = generator.generate(
        item_ids=components["query_ids"],
        **components
    )

    end_time = time.time()
    logger.info(f"Execution time: {end_time - start_time} seconds")
    logger.info(f"Generated {generated_topics} topics.")
    # Save output
    filename = make_file_name(
        data=data,
        model=model,
        prompt=prompt,
        nqueries=nqueries,
        ndocs=ndocs,
        k=k,
    )

    topics = []
    for topic in generated_topics:
        if isinstance(topic, output_class):
            topics.append(topic)
        else:
            logger.warning(f"Error: {topic}")

    generated_topics = Topics[output_class](topics=topics)

    filename += ".jsonl"
    generated_topics.to_jsonl(Path(output) / filename)


if __name__ == "__main__":
    main()
