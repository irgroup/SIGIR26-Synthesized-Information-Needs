from pathlib import Path
import ir_datasets
import pandas as pd


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR_RAW = PROJECT_ROOT / "data" / "raw"
DATA_DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = str(DATA_DIR_RAW / "datasets" / "cache")


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
                    "title": query.title,
                    "description": query.description,
                    "narrative": query.narrative,
                    "uqv": variants,
                    "rel_docs": self.qrels_map.get(query.query_id),
                }
            )

        return self.queries


class ird_qrels_parser:
    def prepare_qrels(dataset_id, k=None):
        def add_doc_text(r):
            doc = store.get(r.doc_id)
            doc_str = doc.title + "\n" + doc.body
            return doc_str[:10000].replace("\n", " ")

        dataset = ir_datasets.load(dataset_id)
        store = dataset.docs_store()

        qrels = pd.DataFrame(dataset.qrels)
        if k:
            qrels = qrels.head(k)
        queries = pd.DataFrame(dataset.queries)

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
