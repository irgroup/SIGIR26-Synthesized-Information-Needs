#!/usr/bin/env python3
"""
Generate TREC Topics

Example usage:
    python scripts/gen-qrels.py --model qwen3-14B-no-think --data robust --k 1 --s --prompt robust-DNA-zero-shot --output .
"""

import click
import time
from pathlib import Path

from topic_gen.generate import Generator
from topic_gen.models import MTO_responds
from topic_gen import logger
from src.data import get_dataset, make_file_name
from src.config import get_llm

logger.setLevel("DEBUG")


@click.command()
# LLM
@click.option("--model", help="The model to use. This must be registered in src/config.py", required=True, type=str)
@click.option("--max_concurrency", help="Maximum number of concurrent requests.", default=50, type=int)
@click.option("--connection", help="The connection string for the LLM.", default="http://localhost:6542/v1", type=str)
@click.option("--gpus", help="CUDA_VISIBLE_DEVICES", default="0", type=str)
# Generation options
@click.option("--data", help="Test collection to be loaded", type=str, required=True)
@click.option("--k", help="Generate results only for the first k samples.", default=None, type=int)
@click.option("--s", help="Generate only the qrels used in the original paper.", is_flag=True, default=False)
@click.option("--prompt", help="Prompt file path", type=click.Path(), required=True)
@click.option("--topics", help="Topics file path", type=click.Path(), required=False, default=None)
@click.option("--output", help="Output file path", type=click.Path(), default=".")
def main(model, max_concurrency, connection, gpus, data, k, s, prompt, topics, output):
    # Get LLM
    llm = get_llm(model, connection=connection, gpus=gpus)

    # load data
    dataset = get_dataset(dataset_name=data)
    components = dataset.qrel_components(k=k, s=s, topics=topics)

    qrels = components.pop("qrels", None)

    # Setup generator
    config = {}
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    generator = Generator(llm=llm, output_class=MTO_responds,
                          prompt=prompt, config=config)

    # Generate qrels
    start_time = time.time()
    res = generator.generate(
        **components
    )
    end_time = time.time()
    logger.info(f"Execution time: {end_time - start_time} seconds")

    # Format result
    llm_judgments = []
    for judgment in res:
        try:
            llm_judgments.append(judgment.O)
        except Exception as e:
            logger.debug(f"Error: {e}")
            llm_judgments.append(999)

    # get original qrels and overwrite the relevance
    qrels["relevance"] = llm_judgments

    # Save output
    filename = make_file_name(
        data=data,
        model=model,
        prompt=prompt,
        topic=topics,
        k=k,
        s=s,
        task="qrels",
    )

    filename += ".csv.gz"
    qrels.to_csv(Path(output) / filename, sep=" ",
                 index=False, header=False, compression="gzip")


if __name__ == "__main__":
    main()
