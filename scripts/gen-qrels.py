#!/usr/bin/env python3
"""
Generate TREC Topics

Example usage:
    python scripts/gen-qrels.py --model qwen3-14B-no-think --data robust --k 1 --prompt robust-DNA-zero-shot --output .
"""
import os
import pandas as pd
import json
from datetime import datetime
import click
import time
from pathlib import Path

from topic_gen.generate import Generator
from topic_gen.models import MTO_responds
from topic_gen import logger
from src.data import get_dataset
from src.config import get_llm
from tirex_tracker import start_tracking, stop_tracking


# logger.setLevel("DEBUG")


@click.command()
# LLM
@click.option("--model", help="The model to use. This must be registered in src/config.py", required=True, type=str)
@click.option("--max_concurrency", help="Maximum number of concurrent requests.", default=50, type=int)
@click.option("--connection", help="The connection string for the LLM.", default="http://localhost:6542/v1", type=str)
# Generation options
@click.option("--data", help="Test collection to be loaded", type=str, required=True)
@click.option("--k", help="Generate results only for the first k samples.", default=None, type=int)
@click.option("--prompt", help="Prompt file path", type=click.Path(), required=True)
@click.option("--topics", help="Topics file path", type=click.Path(), required=False, default=None)
@click.option("--output", help="Output file path", type=click.Path(), default=".")
@click.option("--no_compression", help="Compress output files", is_flag=True, default=False)
@click.option("--all", help="Generate all qrels.", is_flag=True, default=False)
def main(model, max_concurrency, connection, data, k, prompt, topics, output, no_compression, all):
    # setup tracking
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.mkdir(Path(output) / timestamp)
    # handle = start_tracking(export_file_path=Path(
    #     output) / timestamp / "index-ir-metadata.yml")

    # Get LLM
    llm = get_llm(model, connection=connection)

    # load data
    if topics:
        with open(os.path.join(topics, "metadata.json"), "r") as f:
            topics_metadata = json.load(f)
        topics = pd.read_json(os.path.join(
            topics, "topics.jsonl"), lines=True, dtype=str)
        topics.rename(columns={"topic_id": "query_id"}, inplace=True)
    else:
        topics_metadata = {
            "date": timestamp,
            "model": "human",
            "data": data,
            "prompt": "human",
            "k": k,
            "nqueries": None,
            "ndocspos": None,
            "ndocsneg": None,
            "output": output,
            "task": "topics"
        }

    dataset = get_dataset(dataset_name=data)
    components = dataset.qrel_components(k=k, all=all, topics=topics)
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
    with open(Path(output) / timestamp / "metadata.json", "w") as f:
        json.dump({
            "date": timestamp,
            "model": model,
            "data": data,
            "prompt": prompt,
            "k": k,
            "topics": topics_metadata,
            "output": output,
            "task": "qrels",
        }, f)

    if not no_compression:
        qrels.to_csv(Path(output) / timestamp / "qrels.csv.gz", sep=" ",
                     index=False, header=False, compression="gzip")
    else:
        qrels.to_csv(Path(output) / timestamp / "qrels.csv", sep=" ",
                     index=False, header=False)

    # stop_tracking(handle)


if __name__ == "__main__":
    main()
    # main(
    #     [
    #         "--model", "gpt-oss-120B-MT1000",
    #         "--data", "robust",
    #         "--connection", "http://localhost:6543/v1",
    #         "--k", "1",
    #         "--prompt", "-DNA-zero-shot",
    #         "--output", ".",
    #         "--no_compression",
    #     ]
    # )
