# from ir_datasets_longeval import load
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import ir_datasets
import ir_measures
import pandas as pd
from datasets import load_from_disk
from ir_datasets_longeval import load
from pydantic import BaseModel, create_model
from sklearn.model_selection import train_test_split
from topic_gen import logger
from topic_gen.models import BaseTopic, MTO_responds, Topics, TRECTopic


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR_RAW = PROJECT_ROOT / "data" / "raw"
DATA_DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_DIR_INTERIM = PROJECT_ROOT / "data" / "interim"
MODELS_DIR = str(DATA_DIR_RAW / "datasets" / "cache")


class TRECTitle(BaseTopic):
    title: str


class TRECDescription(BaseTopic):
    description: str


class TRECNarrative(BaseTopic):
    narrative: str


class TRECTitleDescription(BaseTopic):
    title: str
    description: str


class TRECTitleNarrative(BaseTopic):
    title: str
    narrative: str


class TRECDescriptionNarrative(BaseTopic):
    description: str
    narrative: str


def alter_class(prompt: str, base_class: BaseModel) -> BaseModel:
    if base_class != TRECTopic:
        raise NotImplementedError(
            "alter_class currently only supports TRECTopic as base_class"
        )

    prompt = prompt.strip(".yaml")
    prompt_splits = prompt.split("-")
    remove_fields = set()
    masked = False
    for field in prompt_splits:
        if field == "masked":
            masked = True
            continue
        if masked:
            remove_fields.add(field)

    # Setup generator
    prompt_name = Path(prompt).stem
    if remove_fields == {"title"}:
        output_class = TRECTitle
    elif remove_fields == {"description", "narrative"}:
        output_class = TRECDescriptionNarrative
    elif remove_fields == {"title", "narrative"}:
        output_class = TRECTitleNarrative
    elif remove_fields == {"title", "description"}:
        output_class = TRECTitleDescription
    elif remove_fields == {"narrative"}:
        output_class = TRECNarrative
    elif remove_fields == {"description"}:
        output_class = TRECDescription
    elif remove_fields == set():
        output_class = TRECTopic
    else:
        raise ValueError(f"Unknown prompt name: {prompt_name}")

    return output_class


class Dataset:
    def __init__(
        self,
        irds: ir_datasets.Dataset,
        topic_class: BaseModel = TRECTopic,
        qrel_class: BaseModel = MTO_responds,
        seed: int = 42,
        n_test_qrels: int = 1000,
    ):
        self.dataset = irds
        self.store = self.dataset.docs_store()
        self.topic_class = topic_class
        self.qrel_class = qrel_class
        self.seed = seed
        self.qrels_test, self.qrels_train = self._split_qrels(n_test_qrels)

    def _sample_query_variants(self, query_id: str, n: int) -> str:
        """Sample n query variants for a given query ID."""
        return ""

    def _split_qrels(self, n_test_qrels) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sample test qrels for evaluation."""
        qrels = pd.DataFrame(self.dataset.qrels)
        qrels_test, qrels_train = train_test_split(
            qrels,
            train_size=n_test_qrels,
            stratify=qrels["relevance"],
            random_state=self.seed,
        )
        return qrels_test, qrels_train

    def _get_document_text(self, doc_id: str) -> str:
        """Retrieve the text of a document given its ID."""
        doc = self.store.get(doc_id)
        return doc.default_text().replace("\n", " ")[:10000].strip()

    def _sample_docs(self, doc_ids: set[str], n: int) -> str:
        """Sample n documents from the given set of document IDs and return their concatenated text."""
        if len(doc_ids) == 0:
            logger.warning(
                "Could not sample Documents: No documents provided.")
            return ""

        sampled_doc_texts = []
        for doc_id in random.sample(doc_ids, min(n, len(doc_ids))):
            sampled_doc_texts.append(self._get_document_text(doc_id))
        joined_docs = "\n\n".join(sampled_doc_texts)
        return joined_docs

    def qrel_components(
        self, k: Optional[int] = None, topics: bool = None, all: bool = False
    ) -> Dict[str, List]:
        """Prepare qrel components for qrel generation."""
        # Load default topics if not provided
        if topics is None:
            topics = pd.DataFrame(self.dataset.queries)

        # Load test qrels if not all queries are requested
        if not all:
            # Ignore train test split and load all qrels from ir_datasets
            qrels = self.qrels_test
        else:
            qrels = pd.DataFrame(self.dataset.qrels)

        # ensure order of columns
        qrels.rename(columns={"iteration": "q0"}, inplace=True)
        qrels.rename(columns={"Q0": "q0"}, inplace=True)
        if "q0" not in qrels.columns:
            qrels["q0"] = "0"
        qrels = qrels[["query_id", "q0", "doc_id", "relevance"]]

        # Remove topics that are not in topics. This is the case if not all topics could be generated
        qrels = qrels[qrels["query_id"].isin(topics["query_id"])]

        # sample k qrels per relevance label
        if k:
            qrels, _ = train_test_split(
                qrels, train_size=k, stratify=qrels["relevance"], random_state=self.seed
            )

        # Add query text
        qrels_extended = qrels.merge(
            topics, left_on="query_id", right_on="query_id")
        # Add document text
        qrels_extended["doc"] = qrels_extended.apply(
            lambda r: self._get_document_text(r.doc_id), axis=1
        )

        # determine query column
        if "title" in qrels_extended.columns:
            query_col = "title"
        elif "text" in qrels_extended.columns:
            query_col = "text"
        else:
            raise ValueError("No title or text column found in topics.")

        if "narrative" not in qrels_extended.columns:
            qrels_extended["narrative"] = ""
        if "description" not in qrels_extended.columns:
            qrels_extended["description"] = ""

        return {
            "document": qrels_extended["doc"].to_list(),
            "query": qrels_extended[query_col].to_list(),
            "narrative": qrels_extended["narrative"].to_list(),
            "description": qrels_extended["description"].to_list(),
            "qrels": qrels,
        }

    def topic_components(
        self, nqueries: int, ndocspos: int, ndocsneg: int, k: Optional[int] = None
    ) -> Dict[str, List]:
        test_query_ids = set(self.qrels_test["query_id"].to_list())

        qids = []
        titles = []
        descriptions = []
        narratives = []
        query_variants = []
        rel_docs = []
        not_rel_docs = []
        notified = False
        for idx, query in enumerate(self.dataset.queries_iter()):
            # Skip topics not in test qrels
            if query.query_id not in test_query_ids:
                continue

            # Title
            if hasattr(query, "title"):
                title = query.title.replace("\n", " ")

            elif hasattr(query, "text"):
                title = query.text.replace("\n", " ")
            else:
                title = ""

            titles.append(title)

            # Description
            if hasattr(query, "description"):
                descriptions.append(query.description.replace("\n", " "))
            else:
                descriptions.append("")

            # Narrative
            if hasattr(query, "narrative"):
                narratives.append(query.narrative.replace("\n", " "))
            else:
                narratives.append("")

            # Query variants
            variants = self._sample_query_variants(query.query_id, nqueries)
            if len(variants) == 0 and not notified:
                logger.warning(
                    f"No query variants available for this dataset. Returning only the original query."
                )
                notified = True
            query_variants.append(title + "\n" + variants)

            # Rel Docs
            rel_doc_ids = self.qrels_train[
                (self.qrels_train["query_id"] == query.query_id)
                & (self.qrels_train["relevance"] > 0)
            ]["doc_id"].to_list()
            rel_docs.append(self._sample_docs(rel_doc_ids, ndocspos))

            # Not rel Docs
            not_rel_doc_ids = self.qrels_train[
                (self.qrels_train["query_id"] == query.query_id)
                & (self.qrels_train["relevance"] == 0)
            ]["doc_id"].to_list()
            not_rel_docs.append(self._sample_docs(not_rel_doc_ids, ndocsneg))

            # Query ID
            qids.append(query.query_id)

            if k is not None and idx + 1 >= k:
                break

        return {
            "query_ids": qids,
            "title": titles,
            "description": descriptions,
            "narrative": narratives,
            "queries": query_variants,
            "relevant_documents": rel_docs,
            "not_relevant_documents": not_rel_docs,
        }


class Robust(Dataset):
    def __init__(self):
        dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
        super().__init__(dataset)

        self.uqv = self._load_uqv()

    def _split_qrels(self, n_test_qrels) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load the original DNA qrels from a parquet file
        def split_ids(qrel_id):
            """Split the qrel_id into query_id and doc_id"""
            qrel_id = qrel_id.strip("-DNA-")
            query_id = qrel_id.split("-")[0]
            doc_id = "-".join(qrel_id.split("-")[1:])
            return query_id, doc_id

        def get_relevance(response):
            response = response.strip()
            if not response.startswith("{"):
                response = "{" + response
            if not response.endswith("}"):
                response = response[: response.rfind("}") + 1]

            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                print(
                    f"Could not decode response: {response}, returning relevance: 0")
                # Only one qrel is left that misses propper key format ("). In this case the relevance is 0
                return 0
            return response.get("O")

        df = pd.read_parquet(DATA_DIR_RAW / "qrel-export.parquet")

        df[["query_id", "doc_id"]] = df["qrel_id"].apply(
            lambda x: pd.Series(split_ids(x))
        )

        # Add a constant column for Q0 as per TREC format
        df["q0"] = "0"

        # drop trec relevance column
        df = df.drop(columns=["relevance"])

        df["relevance"] = df["response"].apply(get_relevance)

        # # Prepare the final qrels DataFrame in TREC format
        qrels_test = df[["query_id", "q0", "doc_id", "relevance"]]

        qrels_train = pd.DataFrame(self.dataset.qrels)
        cols = ["query_id", "doc_id"]
        index_df = qrels_train.set_index(cols).index
        index_df2 = qrels_test.set_index(cols).index

        qrels_train = qrels_train[~index_df.isin(index_df2)]
        return qrels_test, qrels_train

    def _sample_query_variants(self, query_id: str, n: int) -> str:
        """Sample n query variants for a given query ID."""
        random.seed(self.seed)
        variants = self.uqv[self.uqv["qid"] == query_id]["uqv"].to_list()
        variants = random.sample(variants, min(n, len(variants)))
        return "\n".join(variants)

    def _load_uqv(self):
        """Load query variants from UQV dataset."""
        uqv_path = DATA_DIR_RAW / "datasets" / "robust" / "robust-uqv.txt"
        uqv = pd.read_csv(uqv_path, sep=";", names=["query_id", "uqv"])
        uqv["qid"] = uqv["query_id"].apply(lambda x: x.split("-")[0])
        return uqv


class LongEval_C_45(Dataset):
    """Train on the click model qrels and test on the human annotated 45 topics qrel set. Test query IDs determine the topics to be generated."""

    def __init__(self):
        dataset = load("longeval-2023/2022-06/en")
        self.qrels_test_file = "depth_based_45_25_25"
        super().__init__(dataset)

    def _ds_to_qrels(self, ds_path: str):
        """Load Huggingface Datasets dataset and transform it to the TREC qrels format"""
        dataset = load_from_disk(ds_path).to_pandas()
        dataset["Q0"] = "0"
        return dataset[["query_id", "Q0", "doc_id", "relevance"]]

    def _split_qrels(self, n_test_qrels) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load deep qrels as test
        BASEPATH = DATA_DIR_RAW / "datasets" / "long_eval_ictir2025"
        qrels_test = self._ds_to_qrels(BASEPATH / self.qrels_test_file)
        qrels_test["query_id"] = qrels_test["query_id"].apply(
            lambda x: self.dataset.query_id_map[x]
        )
        qrels_test["doc_id"] = qrels_test["doc_id"].apply(
            lambda x: self.dataset.doc_id_map[x]
        )

        # Load click model qrels as train
        qrels_train = pd.DataFrame(self.dataset.qrels)
        return qrels_test, qrels_train


class LongEval_45_C(Dataset):
    """Train on the human annotated 45 topics qrel set. Test on the click model qrels. Test query IDs determine the topics to be generated."""

    def __init__(self):
        dataset = load("longeval-2023/2022-06/en")
        self.qrels_test_file = "depth_based_45_25_25"
        super().__init__(dataset)

    def _ds_to_qrels(self, ds_path: str):
        """Load Huggingface Datasets dataset and transform it to the TREC qrels format"""
        dataset = load_from_disk(ds_path).to_pandas()
        dataset["Q0"] = "0"
        return dataset[["query_id", "Q0", "doc_id", "relevance"]]

    def _split_qrels(self, n_test_qrels) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load deep qrels as test
        BASEPATH = DATA_DIR_RAW / "datasets" / "long_eval_ictir2025"
        qrels_train = self._ds_to_qrels(BASEPATH / self.qrels_test_file)
        qrels_train["query_id"] = qrels_train["query_id"].apply(
            lambda x: self.dataset.query_id_map[x]
        )
        qrels_train["doc_id"] = qrels_train["doc_id"].apply(
            lambda x: self.dataset.doc_id_map[x]
        )

        # Load click model qrels as test
        qrels_test = pd.DataFrame(self.dataset.qrels)
        qrels_test = qrels_test[
            qrels_test["query_id"].isin(set(qrels_train["query_id"]))
        ]
        return qrels_test, qrels_train


class LongEval_45_45(Dataset):
    """Train and test on splits of the human annotated 45 topics qrel set. Test query IDs determine the topics to be generated."""

    def __init__(self):

        dataset = load("longeval-2023/2022-06/en")
        self.qrels_test_file = "depth_based_45_25_25"
        super().__init__(dataset, n_test_qrels=1500)

    def _ds_to_qrels(self, ds_path: str):
        """Load Huggingface Datasets dataset and transform it to the TREC qrels format"""
        dataset = load_from_disk(ds_path).to_pandas()
        dataset["Q0"] = "0"
        return dataset[["query_id", "Q0", "doc_id", "relevance"]]

    def _split_qrels(self, n_test_qrels) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load deep qrels as test
        BASEPATH = DATA_DIR_RAW / "datasets" / "long_eval_ictir2025"
        qrels_test = self._ds_to_qrels(BASEPATH / self.qrels_test_file)
        qrels_test["query_id"] = qrels_test["query_id"].apply(
            lambda x: self.dataset.query_id_map[x]
        )
        qrels_test["doc_id"] = qrels_test["doc_id"].apply(
            lambda x: self.dataset.doc_id_map[x]
        )

        qrels_test, qrels_train = train_test_split(
            qrels_test,
            train_size=n_test_qrels,
            stratify=qrels_test["relevance"],
            random_state=self.seed,
        )
        return qrels_test, qrels_train


class LongEval(Dataset):
    """Train and test on the click model qrels but only for the 45 topics that were annotated by humans. Test query IDs determine the topics to be generated."""

    def __init__(self):
        dataset = load("longeval-2023/2022-06/en")
        super().__init__(dataset, n_test_qrels=100)

    def _ds_to_qrels(self, ds_path: str):
        """Load Huggingface Datasets dataset and transform it to the TREC qrels format"""
        dataset = load_from_disk(ds_path).to_pandas()
        dataset["Q0"] = "0"
        return dataset[["query_id", "Q0", "doc_id", "relevance"]]

    def _split_qrels(self, n_test_qrels) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sample test qrels for evaluation."""
        qrels = pd.DataFrame(self.dataset.qrels)
        self.qrels_test_file = "depth_based_45_25_25"

        BASEPATH = DATA_DIR_RAW / "datasets" / "long_eval_ictir2025"
        qrels_45 = self._ds_to_qrels(BASEPATH / self.qrels_test_file)
        qrels_45["query_id"] = qrels_45["query_id"].apply(
            lambda x: self.dataset.query_id_map[x]
        )
        qrels = qrels[qrels["query_id"].isin(set(qrels_45["query_id"]))]

        return qrels, qrels


class DL19(Dataset):
    def __init__(self):
        dataset = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
        super().__init__(dataset)


def get_dataset(dataset_name: str):
    if dataset_name == "robust":
        return Robust()
    if dataset_name == "dl19":
        return DL19()
    if dataset_name == "longeval-C-45":
        return LongEval_C_45()
    if dataset_name == "longeval-45-C":
        return LongEval_45_C()
    if dataset_name == "longeval-45-45":
        return LongEval_45_45()
    if dataset_name == "longeval":
        return LongEval()
    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented.")
