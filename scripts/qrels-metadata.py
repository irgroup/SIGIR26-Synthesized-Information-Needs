from src.data import DATA_DIR_INTERIM, DATA_DIR_PROCESSED
from src.io import load_qrel_from_path, read_metadata
import json

qrels_metadata = {}
qrels_metadata["qrels_topic_features"] = {}
for dataset in ["dl19", "dl20"]:
    qrels_metadata["qrels_topic_features"][dataset] = {}
    for partial in ["qrels-topics-generated-full", "qrels-topics-generated-title-description", "qrels-topics-generated-title-narrative"]:
                
        input_path = DATA_DIR_INTERIM / dataset / partial
        # input_path = DATA_DIR_INTERIM / "dl20" / "qrels-topics-generated-title-description"
        metadata = read_metadata(input_path)

        df = metadata[(metadata['topics_nqueries'] == 1.0) & (metadata["topics_prompt"].isin(["topic-query", "topic-query-contrastive"]))]
        qrels_metadata["qrels_topic_features"][dataset][partial] =  df["name"].to_list()


with open("qrel_metadata.json", "w") as f:
    json.dump(qrels_metadata, f)