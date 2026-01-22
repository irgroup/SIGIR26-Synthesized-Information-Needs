from pathlib import Path

import pandas as pd
from topic_gen import logger
from topic_gen.evaluate import MetaExperiment
from topic_gen.evaluate.io import load_from_irds
from topic_gen.evaluate.measures_agreement import CohenKappa, MeanAverageError
from topic_gen.evaluate.measures_agreement_multiple import LabelDistribution

from src.data import DATA_DIR_INTERIM, DATA_DIR_PROCESSED
from src.io import load_qrels_from_path, read_metadata

logger.setLevel("DEBUG")


def main(
    dataset: str,
    qrels_path: str,
    baseline: str,
    output: Path,
    binarize_level: int = 0,
):
    experiments, _ = load_qrels_from_path(
        qrels_path,
        binarize_qrels=binarize_level,
        replace_label_mapping={999: 0},
    )
    baseline = load_from_irds(baseline, binarize_level)

    meta_exp = MetaExperiment(
        experiments=experiments,
        baseline=baseline,
        measures=[
            CohenKappa(weights="linear"),
            MeanAverageError(),
            LabelDistribution(),
        ],
        bootstrap=20,
        filter_qrels=True,
        test="t",
        alpha=0.05,
        correction=True,
    )
    res = meta_exp.evaluate()

    df = pd.DataFrame(res)
    metadata = read_metadata(qrels_path, long=True)

    df = pd.concat([df, metadata])

    # fix missing for robust
    if dataset == "robust":
        df.loc[df["measure"] == "missing_qrels", "value"] = abs(
            df.loc[df["measure"] == "missing_qrels", "value"] - 311410 + 2951
        )

    df = df.drop_duplicates()

    if binarize_level > 0:
        binary = "-binary"
    else:
        binary = ""

    df.to_csv(
        output / f"label-alignment-{dataset}-{qrels_path.name}{binary}.tsv",
        index=False,
        sep="\t",
    )


if __name__ == "__main__":
    # Robust
    main(
        dataset="robust",
        qrels_path=DATA_DIR_INTERIM / "robust" / "qrels-topics-generated",
        baseline="disks45/nocr/trec-robust-2004",
        output=DATA_DIR_PROCESSED,
    )

    # main(
    #     dataset="robust",
    #     qrels_path=DATA_DIR_INTERIM / "robust" / "qrels-topics-generated",
    #     baseline="disks45/nocr/trec-robust-2004",
    #     binarize_level=1,
    #     output=DATA_DIR_PROCESSED,
    # )

    # # DL 19
    # main(
    #     dataset="dl19",
    #     qrels_path=DATA_DIR_INTERIM / "dl19" / "qrels-topics-generated-full",
    #     baseline="msmarco-passage/trec-dl-2019/judged",
    #     output=DATA_DIR_PROCESSED,
    # )

    # main(
    #     dataset="dl19",
    #     qrels_path=DATA_DIR_INTERIM / "dl19" / "qrels-topics-generated-full",
    #     baseline="msmarco-passage/trec-dl-2019/judged",
    #     binarize_level=2,
    #     output=DATA_DIR_PROCESSED,
    # )

    # # DL 20
    # main(
    #     dataset="dl20",
    #     qrels_path=DATA_DIR_INTERIM / "dl20" / "qrels-topics-generated-full",
    #     baseline="msmarco-passage/trec-dl-2020/judged",
    #     output=DATA_DIR_PROCESSED,
    # )

    # main(
    #     dataset="dl20",
    #     qrels_path=DATA_DIR_INTERIM / "dl20" / "qrels-topics-generated-full",
    #     baseline="msmarco-passage/trec-dl-2020/judged",
    #     binarize_level=2,
    #     output=DATA_DIR_PROCESSED,
    # )
