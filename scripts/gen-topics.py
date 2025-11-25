#!/usr/bin/env python3
"""
Generate TREC Topics

Example usage:
    python scripts/gen-topics.py --model gpt-oss-120b --k 1 --prompt trec-base

"""
import click
import time
import os
import json
from pathlib import Path
from datetime import datetime

from topic_gen.generate import Generator
from topic_gen.models import Topics
from topic_gen import logger
from src.data import get_dataset, alter_class
from src.config import get_llm
from tirex_tracker import start_tracking, stop_tracking

logger.setLevel("DEBUG")


@click.command()
# LLM
@click.option("--model", help="The model to use. This must be registered in src/config.py", required=True, type=str)
@click.option("--max_concurrency", help="Maximum number of concurrent requests.", default=50, type=int)
@click.option("--connection", help="The connection string for the LLM.", default="http://localhost:6544/v1", type=str)
# Generation options
@click.option("--data", help="Test collection to be loaded", type=str, required=True)
@click.option("--k", help="Generate results only for the first k samples.", default=None, type=int)
@click.option("--prompt", help="Prompt file path", type=click.Path(), required=True)
@click.option("--nqueries", help="Number of query variants.", default=0, type=int)
@click.option("--ndocspos", help="Number of relevant documents.", default=0, type=int)
@click.option("--ndocsneg", help="Number of non-relevant documents.", default=0, type=int)
@click.option("--output", help="Output file path", type=click.Path(), default=".")
def main(model, max_concurrency, connection, data, k, prompt, nqueries, ndocspos, ndocsneg, output):
    # setup tracking
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.mkdir(Path(output) / timestamp)
    handle = start_tracking(export_file_path=Path(
        output) / timestamp / "index-ir-metadata.yml")

    # Get LLM
    llm = get_llm(model, connection=connection)

    # Load data
    dataset = get_dataset(dataset_name=data)

    components = dataset.topic_components(
        k=k, nqueries=nqueries, ndocspos=ndocspos, ndocsneg=ndocsneg)

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
    with open(Path(output) / timestamp / "metadata.json", "w") as f:
        json.dump({
            "date": timestamp,
            "model": model,
            "data": data,
            "prompt": prompt,
            "k": k,
            "nqueries": nqueries,
            "ndocspos": ndocspos,
            "ndocsneg": ndocsneg,
            "output": output,
            "task": "topics"
        }, f)

    topics = []
    for topic in generated_topics:
        if isinstance(topic, output_class):
            topics.append(topic)
        else:
            logger.warning(f"Error: {topic}")

    generated_topics = Topics[output_class](topics=topics)
    generated_topics.to_jsonl(Path(output) / timestamp / "topics.jsonl")

    stop_tracking(handle)


if __name__ == "__main__":
    main()
    # main(
    #     [
    #         "--model", "qwen3-30B-no-think",
    #         "--data", "robust",
    #         "--connection", "http://localhost:6543/v1",
    #         "--k", "1",
    #         "--s",
    #         "--output", ".",
    #         "--prompt", "./data/raw/prompts/trec-contrastive.yaml",
    #     ]
    # )
