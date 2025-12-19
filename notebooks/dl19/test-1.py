
import pandas as pd
from src.data import DATA_DIR_INTERIM
from src.config import LLM_NAMES

from src.io import load_qrels_from_path, read_metadata
from topic_gen.evaluate import MetaExperiment
from topic_gen.evaluate.io import load_from_irds
from topic_gen.evaluate.measures_agreement import CohenKappa, AreaUnderReceiver, MeanAverageError
from topic_gen.evaluate.measures_agreement_multiple import RosKTau, RosRBO, FleissKappa, KrippendorffAlpha

from topic_gen.evaluate.measures_agreement_multiple import LabelDistribution
from topic_gen.evaluate.utils import QrelsTransformer
import ir_datasets

from topic_gen import logger
logger.setLevel("DEBUG")


BASE_DIR = DATA_DIR_INTERIM / "dl19" / "qrels-dl19-reference"

metadata = read_metadata(BASE_DIR)

print("Binary: Drop missing values")
print("===========================================================")

print("Agreement\n-----------------------------------------------------------")
experiments = load_qrels_from_path(
    BASE_DIR, binarize_qrels=2, drop_relevance_values=999)

baseline = load_from_irds(
    "msmarco-passage/trec-dl-2019/judged", binarize_qrels=2)

meta_exp = MetaExperiment(
    experiments=experiments,
    baseline=baseline,
    measures=[CohenKappa(), AreaUnderReceiver(),
              MeanAverageError(), LabelDistribution(), FleissKappa(), KrippendorffAlpha(
        level_of_measurement="nominal")],
    filter_qrels=True
)

res = meta_exp.evaluate()

df = pd.DataFrame(res)
missing = (
    df[["name", "missing_qrels"]].drop_duplicates(
    ).set_index("name").to_dict()["missing_qrels"]
)

df = df.pivot(index="name", columns="measure", values="value").reset_index()
df = df.merge(metadata, left_on="name", right_on="date", how="left")
df["missing"] = df["name"].map(missing)

print(df[["model", "CohenKappa", "MeanAverageError",
          "AreaUnderReceiver", "missing", "label_dist(0)", "label_dist(1)", "label_dist(0)", "label_dist(1)", "FleissKappa", "KrippendorffAlpha"]].round(2))


# print("\n\nBinary: replace missing with 0")
# print("===========================================================")

# BASE_DIR = DATA_DIR_INTERIM / "dl19" / "qrels-dl19-reference"
# experiments = load_qrels_from_path(
#     BASE_DIR, binarize_qrels=2, replace_label_mapping={999: 0})


# baseline = load_from_irds(
#     "msmarco-passage/trec-dl-2019/judged", binarize_qrels=2)


# meta_exp = MetaExperiment(
#     experiments=experiments,
#     baseline=baseline,
#     measures=[CohenKappa(), AreaUnderReceiver(),
#               MeanAverageError(), LabelDistribution()],
#     filter_qrels=True
# )


# res = meta_exp.evaluate()


# metadata = read_metadata(BASE_DIR)


# df = pd.DataFrame(res)
# missing = (
#     df[["name", "missing_qrels"]].drop_duplicates(
#     ).set_index("name").to_dict()["missing_qrels"]
# )

# df = df.pivot(index="name", columns="measure", values="value").reset_index()

# df = df.merge(metadata, left_on="name", right_on="date", how="left")
# df["missing"] = df["name"].map(missing)


# print(df[["name", "model",  "CohenKappa", "MeanAverageError",
#           "AreaUnderReceiver", "missing"]].round(2))


# print("\n\n\n\nGraded")
# print("===========================================================")


# BASE_DIR = DATA_DIR_INTERIM / "dl19" / "qrels-dl19-reference"
# experiments = load_qrels_from_path(BASE_DIR, replace_label_mapping={999: 0})


# baseline = load_from_irds("msmarco-passage/trec-dl-2019/judged")


# meta_exp = MetaExperiment(
#     experiments=experiments,
#     baseline=baseline,
#     measures=[CohenKappa(), MeanAverageError(), LabelDistribution()],
#     filter_qrels=True
# )


# res = meta_exp.evaluate()


# df = pd.DataFrame(res)
# missing = (
#     df[["name", "missing_qrels"]].drop_duplicates(
#     ).set_index("name").to_dict()["missing_qrels"]
# )

# df = df.pivot(index="name", columns="measure", values="value").reset_index()

# df = df.merge(metadata, left_on="name", right_on="date", how="left")
# df["missing"] = df["name"].map(missing)


# print(df[["name", "model", "CohenKappa", "MeanAverageError", "missing",
#           "label_dist(0)", "label_dist(1)", "label_dist(2)", "label_dist(3)",]].round(2))
