#!/usr/bin/env python3
"""
Experiment 2.1: Judge relevance of an existing test collection using the title of the topic only

Example usage:
    python scripts/exp2.1-llm-judge.py --model qwen3-14B-MT100_no_think --k 1000 --connection http://localhost:6544/v1

"""
import click
import time
import os
from topic_gen.generate import Generator
from topic_gen.models import MTO_responds
from src.data import ird_qrels_parser, DATA_DIR_PROCESSED, DATA_DIR_RAW
from topic_gen import logger

from src.config import get_llm

logger.setLevel("DEBUG")


@click.command()
@click.option("--model", help="The model to use. This must be registered in src/config.py", required=True, type=str)
@click.option("--k", help="Generate results only for the first k samples.", default=None, type=int)
@click.option("--s", help="Generate qrels only for the qrels used in the original paper.", is_flag=True, default=False)
@click.option("--max_concurrency", help="Maximum number of concurrent requests.", default=50, type=int)
@click.option("--connection", help="The connection string for the LLM.", default="0", type=str)
@click.option("--gpus", help="CUDA_VISIBLE_DEVICES", default="0", type=str)
def main(model, k, s, max_concurrency, connection, gpus):
    # Get LLM
    llm = get_llm(model, connection=connection, gpus=gpus)

    # Load data
    documents, titles, _, _, qrels = ird_qrels_parser.prepare_qrels(
        "disks45/nocr/trec-robust-2004", k=k, s=s)

    # Setup generator
    prompt = "robust-DNA-zero-shot-masked-description-narrative"
    prompt_file = prompt + ".yaml"
    config = {"return_exceptions": True}
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    generator = Generator(llm=llm, output_class=MTO_responds,
                          prompt=DATA_DIR_RAW / "prompts" / prompt_file,
                          config=config)

    # Generate judgments
    start_time = time.time()

    res = generator.generate(
        document=documents,
        query=titles,
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

    qrels["relevance"] = llm_judgments

    # format filename
    output_file = f"qrels-robust_{model.replace('/', '-')}_{prompt}_topics-trec-title"
    if k is not None:
        output_file += f"_k{k}"
    if s:
        output_file += f"_s"

    output_file += ".csv.gz"

    qrels.to_csv(os.path.join(DATA_DIR_PROCESSED, "exp2",
                 output_file), sep=" ", index=False, header=False, compression="gzip")


if __name__ == "__main__":
    main()
