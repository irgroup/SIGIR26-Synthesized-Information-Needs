from pathlib import Path
import ir_datasets
import pandas as pd
from typing import Optional
import json


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR_RAW = PROJECT_ROOT / "data" / "raw"
DATA_DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = str(DATA_DIR_RAW / "datasets" / "cache")


def parse_qrel_file_names(qrel_file: str) -> tuple:
    """Parses the qrel file name to extract task, model, prompt, topic, and k."""
    qrel_file = qrel_file.strip(".csv.gz")
    qrel_file = qrel_file.strip(".csv")
    task, model, prompt, topic, k = qrel_file.split("_")
    return task, model, prompt, topic, k


class uqv_parser:
    def __init__(self):
        self.queries = []
        self.dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
        self.store = self.dataset.docs_store()

        self.qrels_map = self.make_qrels_map()

    def make_qrels_map(self):
        qrels_map = {}
        for qrel in self.dataset.qrels_iter():
            if not qrels_map.get(qrel.query_id):
                qrels_map[qrel.query_id] = []
            if qrel.relevance > 0:
                doc = self.store.get(qrel.doc_id)
                qrels_map[qrel.query_id].append(doc.body.replace("\n", " "))
        return qrels_map

    def parse_variants(self):
        # variants
        uqv_path = DATA_DIR_RAW / "trec-reference" / "robust-uqv.txt"
        uqv = pd.read_csv(
            uqv_path, sep=";", names=["query_id", "uqv"]
        )
        uqv["qid"] = uqv["query_id"].apply(lambda x: x.split("-")[0])

        for query in self.dataset.queries_iter():
            variants = uqv[uqv["qid"] == query.query_id]["uqv"].to_list()

            self.queries.append(
                {
                    "qid": query.query_id,
                    "title": query.title.replace("\n", " "),
                    "description": query.description.replace("\n", " "),
                    "narrative": query.narrative.replace("\n", " "),
                    "uqv": variants,
                    "rel_docs": self.qrels_map.get(query.query_id),
                }
            )

        return self.queries


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

    df = pd.read_parquet("data/raw/qrel-export.parquet")

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


class ird_qrels_parser:
    def prepare_qrels(dataset_id: str, k: Optional[int] = None, s: bool = False, topics: bool = None):
        def add_doc_text(r):
            doc = store.get(r.doc_id)
            doc_str = doc.title + "\n" + doc.body
            return doc_str[:10000].replace("\n", " ")

        dataset = ir_datasets.load(dataset_id)
        store = dataset.docs_store()

        # Load custom topics file
        if isinstance(topics, pd.DataFrame):
            queries = topics
        else:
            queries = pd.DataFrame(dataset.queries)

        qrels = pd.DataFrame(dataset.qrels)

        # remove potentially unused topics
        qrels = qrels[qrels["query_id"].isin(queries["query_id"])]

        if s:
            # Overwrite qrels with DNA qrels
            qrels = get_DNA_qrels()

        # sample k qrels per relevance label
        if k:
            sampled = []
            for _, group in qrels.groupby("relevance"):
                sampled.append(group.sample(n=k, random_state=42))
            qrels = pd.concat(sampled).sort_index()  # maintain original order

        qrels_extended = qrels.merge(
            queries, left_on="query_id", right_on="query_id")
        qrels_extended["doc"] = qrels_extended.apply(add_doc_text, axis=1)

        documents = qrels_extended["doc"].to_list()
        titles = qrels_extended["title"].to_list()
        narratives = qrels_extended["narrative"].to_list()
        descriptions = qrels_extended["description"].to_list()

        return documents, titles, narratives, descriptions, qrels


def get_dataset(dataset_name):
    if dataset_name == "robust":
        parser = uqv_parser()
        return parser.parse_variants()

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
