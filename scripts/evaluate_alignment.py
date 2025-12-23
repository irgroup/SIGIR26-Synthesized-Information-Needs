import pandas as pd
from topic_gen import logger
from topic_gen.evaluate import MetaExperiment
from topic_gen.evaluate.io import load_from_irds
from topic_gen.evaluate.measures_agreement import (
    AreaUnderReceiver,
    CohenKappa,
    MeanAverageError,
)
from topic_gen.evaluate.measures_agreement_multiple import LabelDistribution

from src.data import DATA_DIR_INTERIM, DATA_DIR_PROCESSED
from src.io import load_qrels_from_path, read_metadata

logger.setLevel("DEBUG")

BASELINE = {
    "robust": "disks45/nocr/trec-robust-2004",
}


def main(dataset, input_):
    BASE_DIR = DATA_DIR_INTERIM / dataset / input_
    # experiments = load_qrels_from_path(
    #     BASE_DIR, binarize_qrels=1, drop_relevance_values=999
    # )
    experiments = load_qrels_from_path(
        BASE_DIR, binarize_qrels=1, replace_label_mapping={999: 0}
    )
    baseline = load_from_irds(BASELINE[dataset], binarize_qrels=1)

    meta_exp = MetaExperiment(
        experiments=experiments,
        baseline=baseline,
        measures=[
            CohenKappa(),
            AreaUnderReceiver(),
            MeanAverageError(),
            LabelDistribution(),
        ],
        bootstrap=20,
        filter_qrels=True,
    )

    res = meta_exp.evaluate()

    metadata = read_metadata(BASE_DIR)

    df = pd.DataFrame(res)

    missing = (
        df.groupby("name")["missing_qrels"].max().to_dict()
    )  # we take the max per group because label dist does not report missings currently

    ci_lower = df.groupby(["name", "measure"])["ci_lower"].max().to_dict()
    ci_upper = df.groupby(["name", "measure"])["ci_upper"].max().to_dict()
    
    df = df.pivot(index="name", columns="measure", values="value").reset_index()
    df = df.merge(metadata, left_on="name", right_on="date")
    for measure in ci_lower.keys():
        df.loc[
            df["name"] == measure[0], f"{measure[1]}_ci_lower"
        ] = ci_lower[measure]
        df.loc[
            df["name"] == measure[0], f"{measure[1]}_ci_upper"
        ] = ci_upper[measure]
        df.loc[df["name"] == measure[0], f"{measure[1]}_ci"] = (
            (ci_upper[measure] - ci_lower[measure]) / 2
        )

    df["missing"] = df["name"].map(missing)
    df["missing"] = abs(df["missing"] - 311410 + 2951)

    df = df.rename(columns={"name": "qrels_id", "date": "topics_id"})

    df.to_csv(
        DATA_DIR_PROCESSED / f"alignment-{dataset}-{input_}.tsv",
        index=False,
        sep="\t",
    )


if __name__ == "__main__":
    dataset = "robust"
    input_ = "qrels-topics-masked"
    main(dataset, input_)
