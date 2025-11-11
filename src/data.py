import os
from pathlib import Path
import ir_datasets
import pandas as pd
from typing import Optional, Union, List, Dict
import json
from topic_gen.models import MTO_responds, TRECTopic, BaseTopic
import random
from pydantic import BaseModel, create_model
from topic_gen import logger


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR_RAW = PROJECT_ROOT / "data" / "raw"
DATA_DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = str(DATA_DIR_RAW / "datasets" / "cache")


def parse_file_names(filename: str) -> tuple:
    """Parses the qrel file name to extract task, model, prompt, topic, and k."""
    filename = filename.strip(".csv.gz")
    filename = filename.strip(".csv")
    filename_parts = filename.split("_")
    task = filename_parts[0]
    data = filename_parts[1]
    model = filename_parts[2]
    prompt = filename_parts[3]
    topics = filename_parts[4]
    extra = filename_parts[5:]
    return task, data, model, prompt, topics, extra


def make_file_name(data: str, model: str, prompt: str, topic: Optional[str] = None, nqueries: Optional[int] = None, ndocs: Optional[int] = None, k: Optional[int] = None, s: Optional[int] = None, task: Optional[str] = "topics") -> str:
    model = model.replace("/", "-")
    # ensure prompt is filename
    prompt = os.path.splitext(os.path.basename(prompt))[0]
    filename = f"{task}_{data}_{model}_{prompt}"
    if task == "qrel":
        if topic:
            topic = os.path.splitext(os.path.basename(topic))[0]
            topic = topic.replace("_", "-")
        else:
            topic = "topics-trec"
        filename += f"_{topic}"

    if nqueries is not None:
        filename += f"_nq{nqueries}"
    if ndocs is not None:
        filename += f"_nd{ndocs}"
    if k is not None:
        filename += f"_k{k}"
    if s is not None:
        filename += f"_s"
    return filename


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


class Dataset:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.topic_class = TRECTopic
        self.qrel_class = MTO_responds

    def topic_components(self) -> Dict[str, Union[pd.DataFrame, List]]:
        """Return the components needed to generate a topic."""
        return {"qids": None,
                "title": None,
                "description": None,
                "narrative": None,
                "queries": None,
                "relevant_documents": None}

    def qrel_components(self) -> Dict[str, Union[pd.DataFrame, List]]:
        """Return the components needed to generate qrels."""
        return {"document": None,
                "queries": None,
                "narrative": None,
                "description": None,
                "qrels": None}


class Robust(Dataset):
    def __init__(self):
        self.dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
        super().__init__("robust")
        self.store = self.dataset.docs_store()
        self.qrels_map = self._make_qrels_map()

    def _make_qrels_map(self):
        """Construct a mapping from query IDs to their relevant documents."""
        qrels_map = {}
        for qrel in self.dataset.qrels_iter():
            if not qrels_map.get(qrel.query_id):
                qrels_map[qrel.query_id] = []
            if qrel.relevance > 0:
                doc = self.store.get(qrel.doc_id)
                qrels_map[qrel.query_id].append(
                    doc.body.replace("\n", " ")[:10000])
        return qrels_map

    def get_rel_docs(self, query_id: str) -> list:
        """Get relevant documents for a given query ID."""
        return self.qrels_map.get(query_id, [])

    def _load_query_variants(self):
        """Load query variants from UQV dataset."""
        uqv_path = DATA_DIR_RAW / "trec-reference" / "robust-uqv.txt"
        uqv = pd.read_csv(
            uqv_path, sep=";", names=["query_id", "uqv"]
        )
        uqv["qid"] = uqv["query_id"].apply(lambda x: x.split("-")[0])
        return uqv

    def topic_components(self, seed: Optional[int] = 42, k: Optional[int] = None, s: Optional[bool] = False, sample_queries: Optional[int] = None, sample_docs: Optional[int] = None) -> Dict[str, List]:
        """Return the components of the topics as lists."""
        random.seed(seed)
        if s:
            qrels = get_DNA_qrels()
            qrels = set(qrels["query_id"].to_list())

        qids = []
        titles = []
        descriptions = []
        narratives = []
        query_variants = []
        rel_docs = []

        uqv = self._load_query_variants()
        for idx, query in enumerate(self.dataset.queries_iter()):
            if s and query.query_id not in qrels:
                continue
            # Query variants
            variants = uqv[uqv["qid"] == query.query_id]["uqv"].to_list()
            if sample_queries is not None:
                variants = random.sample(
                    variants, min(sample_queries, len(variants)))
            query_variants.append("\n".join(variants))

            # Rel Docs
            all_rel_docs = self.qrels_map.get(query.query_id)
            if not all_rel_docs:
                print(
                    f"Warning: Topic {query.query_id} has no relevant documents.")
                continue
            if sample_docs is not None:
                all_rel_docs = random.sample(
                    all_rel_docs, min(sample_docs, len(all_rel_docs)))
            rel_docs.append("\n\n".join(all_rel_docs))

            # Other components
            qids.append(query.query_id)
            titles.append(query.title.replace("\n", " "))
            descriptions.append(query.description.replace("\n", " "))
            narratives.append(query.narrative.replace("\n", " "))

            if k is not None and idx + 1 >= k:
                break

        return {
            "query_ids": qids,
            "title": titles,
            "description": descriptions,
            "narrative": narratives,
            "queries": query_variants,
            "relevant_documents": rel_docs,
        }

    def qrel_components(self, k: Optional[int] = None, s: bool = False, topics: bool = None) -> Dict[str, List]:
        def add_doc_text(r):
            doc = self.store.get(r.doc_id)
            doc_str = doc.title + "\n" + doc.body
            return doc_str[:10000].replace("\n", " ")

        # Load custom topics file
        if isinstance(topics, pd.DataFrame):
            queries = topics
        elif isinstance(topics, str):
            logger.info(f"Loading custom topics file from {topics}")
            queries = pd.read_json(topics, lines=True, dtype=str)
            queries.rename(columns={"topic_id": "query_id"}, inplace=True)
        else:
            queries = pd.DataFrame(self.dataset.queries)

        qrels = pd.DataFrame(self.dataset.qrels)

        if s:
            # Overwrite qrels with DNA qrels
            qrels = get_DNA_qrels()

        # remove potentially unused topics
        qrels = qrels[qrels["query_id"].isin(queries["query_id"])]

        # sample k qrels per relevance label
        if k:
            sampled = []
            for _, group in qrels.groupby("relevance"):
                sampled.append(group.sample(n=k, random_state=42))
            qrels = pd.concat(sampled).sort_index()  # maintain original order

        qrels_extended = qrels.merge(
            queries, left_on="query_id", right_on="query_id")
        qrels_extended["doc"] = qrels_extended.apply(add_doc_text, axis=1)

        return {
            "document": qrels_extended["doc"].to_list(),
            "query": qrels_extended["title"].to_list(),
            "narrative": qrels_extended["narrative"].to_list(),
            "description": qrels_extended["description"].to_list(),
            "qrels": qrels,
        }


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


# class uqv_parser:
#     def __init__(self):
#         self.queries = []
#         self.dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
#         self.store = self.dataset.docs_store()

#         self.qrels_map = self.make_qrels_map()

#     def make_qrels_map(self):
#         qrels_map = {}
#         for qrel in self.dataset.qrels_iter():
#             if not qrels_map.get(qrel.query_id):
#                 qrels_map[qrel.query_id] = []
#             if qrel.relevance > 0:
#                 doc = self.store.get(qrel.doc_id)
#                 qrels_map[qrel.query_id].append(doc.body.replace("\n", " "))
#         return qrels_map

#     def parse_variants(self):
#         # variants
#         uqv_path = DATA_DIR_RAW / "trec-reference" / "robust-uqv.txt"
#         uqv = pd.read_csv(
#             uqv_path, sep=";", names=["query_id", "uqv"]
#         )
#         uqv["qid"] = uqv["query_id"].apply(lambda x: x.split("-")[0])

#         for query in self.dataset.queries_iter():
#             variants = uqv[uqv["qid"] == query.query_id]["uqv"].to_list()

#             self.queries.append(
#                 {
#                     "qid": query.query_id,
#                     "title": query.title.replace("\n", " "),
#                     "description": query.description.replace("\n", " "),
#                     "narrative": query.narrative.replace("\n", " "),
#                     "uqv": variants,
#                     "rel_docs": self.qrels_map.get(query.query_id),
#                 }
#             )

#         return self.queries


# class ird_qrels_parser:
#     def prepare_qrels(dataset_id: str, k: Optional[int] = None, s: bool = False, topics: bool = None):
#         def add_doc_text(r):
#             doc = store.get(r.doc_id)
#             doc_str = doc.title + "\n" + doc.body
#             return doc_str[:10000].replace("\n", " ")

#         dataset = ir_datasets.load(dataset_id)
#         store = dataset.docs_store()

#         # Load custom topics file
#         if isinstance(topics, pd.DataFrame):
#             queries = topics
#         else:
#             queries = pd.DataFrame(dataset.queries)

#         qrels = pd.DataFrame(dataset.qrels)

#         # remove potentially unused topics
#         qrels = qrels[qrels["query_id"].isin(queries["query_id"])]

#         if s:
#             # Overwrite qrels with DNA qrels
#             qrels = get_DNA_qrels()

#         # sample k qrels per relevance label
#         if k:
#             sampled = []
#             for _, group in qrels.groupby("relevance"):
#                 sampled.append(group.sample(n=k, random_state=42))
#             qrels = pd.concat(sampled).sort_index()  # maintain original order

#         qrels_extended = qrels.merge(
#             queries, left_on="query_id", right_on="query_id")
#         qrels_extended["doc"] = qrels_extended.apply(add_doc_text, axis=1)

#         documents = qrels_extended["doc"].to_list()
#         titles = qrels_extended["title"].to_list()
#         narratives = qrels_extended["narrative"].to_list()
#         descriptions = qrels_extended["description"].to_list()

#         return documents, titles, narratives, descriptions, qrels


def get_dataset(dataset_name: str):
    if dataset_name == "robust":
        return Robust()
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
