import os

import pandas as pd
from topic_gen import logger
from topic_gen.evaluate import Experiment, MetaExperiment
from topic_gen.evaluate.io import read_irds_topics
from topic_gen.evaluate.measures_agreement import (
    BertScore,
    CosineSimilarity,
    JaccardIndex,
    RelativeLength,
    RougeScore,
)
from topic_gen.evaluate.utils import TopicsTransformer

from src.data import DATA_DIR_INTERIM, DATA_DIR_PROCESSED
from src.io import load_topics_from_path, read_metadata

logger.setLevel("DEBUG")

BASELINE = {
    "robust": "disks45/nocr/trec-robust-2004",
}


def main(dataset, input_):
    BASE_DIR = DATA_DIR_INTERIM / dataset / input_

    # experiments = []
    # metadata = []
    # for model in os.listdir(BASE_DIR):
    #     if model == "Llama3.3-70B":
    #         continue
    #     print(model)
    #     metadata.append(read_metadata(BASE_DIR / model, long=True))
    #     experiments.extend(load_topics_from_path(BASE_DIR / model))
    # metadata = pd.concat(metadata)

    experiments = load_topics_from_path(BASE_DIR)
    metadata = read_metadata(BASE_DIR , long=True)

    baseline = Experiment(topics=read_irds_topics("disks45/nocr/trec-robust-2004"))

    # create a combined field
    experiments = [TopicsTransformer.add_combined_field(exp) for exp in experiments]
    baseline = TopicsTransformer.add_combined_field(baseline)

    meta_exp = MetaExperiment(
        experiments=experiments,
        baseline=baseline,
        # measures=[JaccardIndex(), RelativeLength(), CosineSimilarity()],
        filter_topics=True,
        # bootstrap=20
        measures=[
            RougeScore(),
            JaccardIndex(),
            BertScore(),
            RelativeLength(),
            CosineSimilarity(),
        ],
    )

    res = meta_exp.evaluate()

    df = pd.DataFrame(res)

    df = pd.concat([df, metadata])
    print(df)

    df.to_csv(
        DATA_DIR_PROCESSED / f"topic-similarity-{dataset}-{input_}.tsv",
        index=False,
        sep="\t",
    )


if __name__ == "__main__":
    dataset = "robust"
    input_ = "topics-q1"
    input_ = "topics-masked"
    main(dataset, input_)
