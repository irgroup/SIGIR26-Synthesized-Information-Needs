import ir_datasets
import pandas as pd


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
        uqv = pd.read_csv(
            "data/raw/trec-reference/robust-uqv.txt", sep=";", names=["query_id", "uqv"]
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


def get_dataset(dataset_name):
    if dataset_name == "robust":
        parser = uqv_parser()
        return parser.parse_variants()

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
