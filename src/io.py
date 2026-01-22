import json
import os
from pathlib import Path
from typing import Dict, List, Union

import ir_measures
import pandas as pd
from topic_gen import logger
from topic_gen.evaluate import Experiment, MeasureResult
from topic_gen.evaluate.io import read_jsonl_topics
from topic_gen.evaluate.utils import QrelsTransformer

from src.config import LLM_NAMES


def load_topics_from_path(topics_path: Union[str, Path]) -> List[Experiment]:
    experiments = []
    for result in os.listdir(topics_path):
        if not os.path.isdir(os.path.join(topics_path, result)):
            continue
        try:
            exp = Experiment(
                topics=read_jsonl_topics(topics_path / result / "topics.jsonl"),
                name=result,
            )
            experiments.append(exp)
        except Exception as e:
            print(f"Error loading experiment from {result}: {e}")
            continue
    return experiments


def read_metadata(path: Path, long: bool = False) -> pd.DataFrame:
    metadata_records = []
    for result in os.listdir(path):
        if not os.path.isdir(os.path.join(path, result)):
            continue
        try:
            with open(os.path.join(path, result, "metadata.json")) as f:
                metadata = json.load(f)

        except FileNotFoundError:
            logger.warning(f"Metadata not found for result {result}, skipping...")
            continue

        metadata["model"] = LLM_NAMES.get(metadata["model"], metadata["model"])
        metadata["prompt"] = str(Path(metadata["prompt"]).stem)
        metadata_records.append(metadata)

    metadata = pd.DataFrame(metadata_records)
    if "topics" in metadata.columns:
        metadata = metadata.join(
            pd.json_normalize(metadata["topics"]).add_prefix("topics_")
        )
        metadata.drop(columns=["topics"], inplace=True)
        metadata["topics_prompt"] = metadata["topics_prompt"].apply(
            lambda p: str(Path(p).stem) if pd.notnull(p) else "human"
        )
        metadata["topics_model"] = metadata["topics_model"].apply(LLM_NAMES.get)

        metadata = metadata.rename(columns={"date": "name"})
        if long:
            return metadata.melt(id_vars="name", var_name="measure", value_name="value")

    if long:
        metadata = metadata.rename(columns={"date": "name"})
        return metadata.melt(id_vars="name", var_name="measure", value_name="value")

    return metadata


def load_qrel_from_path(
    qrels_path: Path,
    binarize_qrels: int = 0,
    replace_label_mapping: Dict[int, int] = None,
    drop_relevance_values: int = None,
) -> Experiment:
    qrels = ir_measures.read_trec_qrels(os.path.join(qrels_path, "qrels.csv.gz"))

    if replace_label_mapping:
        qrels, missing = QrelsTransformer.replace_relevance(
            qrels, replace_label_mapping
        )
    if drop_relevance_values:
        qrels, missing = QrelsTransformer.drop_relevance(
            qrels, drop_values=drop_relevance_values
        )

    exp = Experiment(
        qrels=qrels,
        name=qrels_path.stem,
        binarize_qrels=binarize_qrels,
    )
    missing_values = MeasureResult(name=qrels_path.stem, measure="missing_qrels_load", value=missing)
    return exp, missing_values


def load_qrels_from_path(
    qrels_path: Union[str, Path],
    binarize_qrels: int = 0,
    replace_label_mapping: Dict[int, int] = None,
    drop_relevance_values: int = None,
) -> List[Experiment]:
    experiments = []
    missing_values = []
    for result in os.listdir(qrels_path):
        if not os.path.isdir(os.path.join(qrels_path, result)):
            continue
        try:
            qrels = ir_measures.read_trec_qrels(
                os.path.join(qrels_path, result, "qrels.csv.gz")
            )
            missing = 0
            if replace_label_mapping:
                qrels, missing = QrelsTransformer.replace_relevance(
                    qrels, replace_label_mapping
                )
            if drop_relevance_values:
                qrels, missing = QrelsTransformer.drop_relevance(
                    qrels, drop_values=drop_relevance_values
                )

            if len(qrels.keys()) == 0:
                logger.warning(
                    f"Qrels for result {result} is empty after processing, skipping..."
                )
                continue

            exp = Experiment(
                qrels=qrels,
                name=result,
                binarize_qrels=binarize_qrels,
            )

            missing_values.append(
                MeasureResult(name=result, measure="missing_qrels_load", value=missing)
            )
            experiments.append(exp)
        except Exception as e:
            print(f"Error loading experiment from {result}: {e}")
            continue
    return experiments, missing_values
