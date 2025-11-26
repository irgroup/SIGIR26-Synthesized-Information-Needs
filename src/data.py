# from ir_datasets_longeval import load
from ir_datasets_longeval import load
import os
from pathlib import Path
import ir_datasets
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
import json
from topic_gen.models import MTO_responds, TRECTopic, BaseTopic
import random
from pydantic import BaseModel, create_model
from topic_gen import logger
import ir_measures


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR_RAW = PROJECT_ROOT / "data" / "raw"
DATA_DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_DIR_INTERIM = PROJECT_ROOT / "data" / "interim"
MODELS_DIR = str(DATA_DIR_RAW / "datasets" / "cache")


def metadata_to_table(metadata_records: List[Dict]) -> pd.DataFrame:
    # metadata table
    metadata = pd.DataFrame(metadata_records)
    metadata = metadata.join(pd.json_normalize(
        metadata["topics"]).add_prefix("topics_"))
    metadata.drop(columns=["topics"], inplace=True)
    metadata["topics_prompt"] = metadata["topics_prompt"].apply(
        lambda p: str(Path(p).stem) if pd.notnull(p) else "human")
    metadata["prompt"] = metadata["prompt"].apply(lambda p: str(Path(p).stem))
    metadata["model"] = metadata["model"].str.replace("-MT1000", "")
    metadata["model"] = metadata["model"].str.replace("-MT100", "")
    return metadata


def load_qrels_from_path(qrels_path: Union[str, Path]) -> Tuple[List[pd.DataFrame], List[str], List[Dict]]:
    predictions = []
    names = []
    metadata_records = []

    for result in os.listdir(qrels_path):
        if not os.path.isdir(os.path.join(qrels_path, result)):
            continue

        # metadata
        try:
            with open(os.path.join(qrels_path, result, "metadata.json")) as f:
                metadata = json.load(f)
            metadata_records.append(metadata)
        except FileNotFoundError:
            logger.warning(
                f"Metadata not found for result {result}, skipping...")
            continue
        # predictions
        qrels = ir_measures.read_trec_qrels(
            os.path.join(qrels_path, result, "qrels.csv.gz"))
        predictions.append(qrels)
        # names
        names.append(result)

    return predictions, names, metadata_to_table(metadata_records)


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
            "alter_class currently only supports TRECTopic as base_class")

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


def get_DNA_qrels():
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

    df = pd.read_parquet(DATA_DIR_RAW/"qrel-export.parquet")

    df[["query_id", "doc_id"]] = df["qrel_id"].apply(
        lambda x: pd.Series(split_ids(x)))

    # Add a constant column for Q0 as per TREC format
    df["q0"] = "0"

    # drop trec relevance column
    df = df.drop(columns=["relevance"])

    df["relevance"] = df["response"].apply(get_relevance)

    # # Prepare the final qrels DataFrame in TREC format
    qrels = df[["query_id", "q0", "doc_id", "relevance"]]
    return qrels


class Dataset:
    def __init__(self, irds: ir_datasets.Dataset, topic_class: BaseModel = TRECTopic, qrel_class: BaseModel = MTO_responds, seed: int = 42, n_test_qrels: int = 1000):
        self.dataset = irds
        self.store = self.dataset.docs_store()
        self.topic_class = topic_class
        self.qrel_class = qrel_class
        self.seed = seed
        self.n_test_qrels = n_test_qrels

    def _sample_query_variants(self, query_id: str, n: int) -> str:
        """Sample n query variants for a given query ID."""
        return ""

    def _sample_test_qrels(self) -> pd.DataFrame:
        """Sample test qrels for evaluation."""
        qrels = pd.DataFrame(self.dataset.qrels)
        sampled = []
        for _, group in qrels.groupby("relevance"):
            if len(group) < self.n_test_qrels:
                relevance_label = group['relevance'].iloc[0]
                logger.warning(
                    f"Not enough qrels for relevance {relevance_label}. Requested {self.n_test_qrels}, but only {len(group)} available. Using all available qrels.")
            max_n = min(len(group), self.n_test_qrels)
            sampled.append(group.sample(n=max_n, random_state=self.seed))

        return pd.concat(sampled).sort_index()

    def _get_document_text(self, doc_id: str) -> str:
        """Retrieve the text of a document given its ID."""
        doc = self.store.get(doc_id)
        return doc.default_text().replace("\n", " ")[:10000].strip()

    def _sample_docs(self, doc_ids: set[str], n: int) -> str:
        """Sample n documents from the given set of document IDs and return their concatenated text."""
        if len(doc_ids) == 0:
            logger.warning("No document IDs provided for sampling.")
            return ""

        sampled_doc_texts = []
        for doc_id in random.sample(doc_ids, min(n, len(doc_ids))):
            sampled_doc_texts.append(self._get_document_text(doc_id))
        joined_docs = "\n\n".join(sampled_doc_texts)
        return joined_docs

    def qrel_components(self, k: Optional[int] = None, topics: bool = None, all: bool = False) -> Dict[str, List]:
        # Load defould topics if not provided
        if topics is None:
            topics = pd.DataFrame(self.dataset.queries)

        # Load test qrels if not all queries are requested
        if not all:
            qrels = self._sample_test_qrels()
        else:
            qrels = pd.DataFrame(self.dataset.qrels)

        # Remove topics that are not judged
        qrels = qrels[qrels["query_id"].isin(topics["query_id"])]

        # sample k qrels per relevance label
        if k:
            sampled = []
            for _, group in qrels.groupby("relevance"):
                sampled.append(group.sample(n=k, random_state=42))
            qrels = pd.concat(sampled).sort_index()

        # load components
        qrels_extended = qrels.merge(
            topics, left_on="query_id", right_on="query_id")
        qrels_extended["doc"] = qrels_extended.apply(
            lambda r: self._get_document_text(r.doc_id), axis=1)

        if "narrative" not in qrels_extended.columns:
            qrels_extended["narrative"] = ""
        if "description" not in qrels_extended.columns:
            qrels_extended["description"] = ""

        return {
            "document": qrels_extended["doc"].to_list(),
            "query": qrels_extended["title"].to_list(),
            "narrative": qrels_extended["narrative"].to_list(),
            "description": qrels_extended["description"].to_list(),
            "qrels": qrels,
        }

    def topic_components(self, nqueries: int, ndocspos: int, ndocsneg: int, k: Optional[int] = None) -> Dict[str, List]:
        qrels = pd.DataFrame(self.dataset.qrels)

        # remove sampled qrels from original qrels to avoid traioning on judged docs
        test_qrels = self._sample_test_qrels()
        cols = ["query_id", "doc_id"]
        index_df = qrels.set_index(cols).index
        index_df2 = test_qrels.set_index(cols).index

        qrels = qrels[~index_df.isin(index_df2)]

        test_query_ids = set(test_qrels["query_id"].to_list())

        qids = []
        titles = []
        descriptions = []
        narratives = []
        query_variants = []
        rel_docs = []
        not_rel_docs = []

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
            query_variants.append(title + "\n" + variants)

            # Rel Docs
            rel_doc_ids = qrels[(qrels["query_id"] == query.query_id) & (
                qrels["relevance"] > 0)]["doc_id"].to_list()
            rel_docs.append(self._sample_docs(rel_doc_ids, ndocspos))

            # Not rel Docs
            not_rel_doc_ids = qrels[(qrels["query_id"] == query.query_id) & (
                qrels["relevance"] == 0)]["doc_id"].to_list()
            not_rel_docs.append(self._sample_docs(
                not_rel_doc_ids, ndocsneg))

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

    def _sample_test_qrels(self) -> pd.DataFrame:
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

        df = pd.read_parquet(DATA_DIR_RAW/"qrel-export.parquet")

        df[["query_id", "doc_id"]] = df["qrel_id"].apply(
            lambda x: pd.Series(split_ids(x)))

        # Add a constant column for Q0 as per TREC format
        df["q0"] = "0"

        # drop trec relevance column
        df = df.drop(columns=["relevance"])

        df["relevance"] = df["response"].apply(get_relevance)

        # # Prepare the final qrels DataFrame in TREC format
        qrels = df[["query_id", "q0", "doc_id", "relevance"]]
        return qrels

    def _sample_query_variants(self, query_id: str, n: int) -> str:
        """Sample n query variants for a given query ID."""
        random.seed(self.seed)
        variants = self.uqv[self.uqv["qid"] == query_id]["uqv"].to_list()
        variants = random.sample(variants, min(n, len(variants)))
        return "\n".join(variants)

    def _load_uqv(self):
        """Load query variants from UQV dataset."""
        uqv_path = DATA_DIR_RAW / "datasets" / "robust" / "robust-uqv.txt"
        uqv = pd.read_csv(
            uqv_path, sep=";", names=["query_id", "uqv"]
        )
        uqv["qid"] = uqv["query_id"].apply(lambda x: x.split("-")[0])
        return uqv


class LongEval(Dataset):
    def __init__(self):
        dataset = load("longeval-2023/2022-09/en")
        super().__init__(dataset)


class DL22(Dataset):
    def __init__(self):
        dataset = ir_datasets.load("msmarco-document-v2/trec-dl-2022/judged")
        super().__init__(dataset)


def get_dataset(dataset_name: str):
    if dataset_name == "robust":
        return Robust()
    if dataset_name == "longeval":
        return LongEval()
    if dataset_name == "trec-dl-2022":
        return DL22()
    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented.")
