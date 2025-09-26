#!/usr/bin/env python3
"""
Experiment 2.1: Judge relevance of an existing test collection using the title of the topic only

Example usage:
    python scripts/exp2.1-llm-judge.py --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --k 50 --gpus 1

"""
import click
import time
import os
import torch
from langchain.chat_models import init_chat_model
from topic_gen.generate import Generator
from topic_gen.models import MTO_responds
from langchain_community.llms import VLLM
from src.data import MODELS_DIR, ird_qrels_parser, DATA_DIR_PROCESSED, DATA_DIR_RAW
from topic_gen import logger
from langchain_openai import ChatOpenAI

logger.setLevel("DEBUG")


@click.command()
@click.option("--model", help="The model to use.")
@click.option("--k", help="Generate results only for the first k samples.", default=None, type=int)
@click.option("--max_concurrency", help="Maximum number of concurrent requests.", default=50, type=int)
@click.option("--gpus", help="CUDA_VISIBLE_DEVICES", default="0", type=str)
def main(model, k, max_concurrency, gpus):
    logger.info(f"Cuda: {torch.cuda.is_available()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

    # Setup the model
    llm = ChatOpenAI(
        openai_api_base="http://localhost:6543/v1",
        openai_api_key="not-needed",
        model_name=model
    )
    # llm = VLLM(model=model,
    #            download_dir=MODELS_DIR,
    #            temperature=0.7,
    #            top_p=0.8,
    #            top_k=20,
    #            min_p=0,
    #            vllm_kwargs={
    #                "max_model_len": 32768,
    #                "max_num_batched_tokens": 32768,
    #                "gpu_memory_utilization": 0.7
    #            }
    #            )

    # Load data
    documents, titles, _, _, qrels = ird_qrels_parser.prepare_qrels(
        "disks45/nocr/trec-robust-2004", k=k)

    # Setup generator
    prompt_name = "robust-DNA-zero-shot-no-title-description"
    generator = Generator(llm=llm, output_class=MTO_responds,
                          prompts_dir=DATA_DIR_RAW / "prompts",
                          prompt_name=prompt_name)

    # Generate judgments
    start_time = time.time()
    config = {"return_exceptions": True}
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency

    res = generator.generate(
        document=documents,
        query=titles,
        config=config
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
    output_file = f"qrels-robust-{model.replace('/', '-')}-{prompt_name}"
    if k is not None:
        output_file += f"-k{k}"
    output_file += ".csv.gz"

    qrels.to_csv(os.path.join(DATA_DIR_PROCESSED,
                 output_file), sep=" ", index=False, header=False, compression="gzip")


if __name__ == "__main__":
    main()
