import pandas as pd

from src.config import MODEL_SORTER, COMPONENTS_SORTER
from src.data import DATA_DIR_PROCESSED


def prompt_to_components(prompt: str) -> dict:
    parts = prompt.split("-")
    components = []
    if "title" not in parts:
        components.append("Title")
    if "description" not in parts:
        components.append("Description")
    if "narrative" not in parts:
        components.append("Narrative")
    if prompt == "-DNA-zero-shot":
        return "Title, Description, Narrative"
    return ", ".join(components)


def main(dataset, input_):
    df = pd.read_csv(DATA_DIR_PROCESSED / f"alignment-{dataset}-{input_}.tsv", sep="\t")
    df = df[df["name"] != "disks45/nocr/trec-robust-2004"]

    df = df.pivot(index="name", columns="measure", values="value")
    df.columns.name = None
    df = df.reset_index()

    # df = df.dropna(subset=["Cohens $\\kappa$"])
    df["components"] = df["prompt"].apply(prompt_to_components)
    df["components"] = pd.Categorical(df["components"], COMPONENTS_SORTER)

    df["model"] = pd.Categorical(df["model"], MODEL_SORTER)

    df = df[
        [
            "model",
            "components",
            "Cohens $\\kappa$",
            "MAE",
            "AUR",
            "missing_qrels_load",
            "AUR_ci",
            "Cohens $\\kappa$_ci",
            "MAE_ci",
        ]
    ]
    # convert to floats
    df["Cohens $\\kappa$"] = df["Cohens $\\kappa$"].astype(float)
    df["MAE"] = df["MAE"].astype(float)
    df["AUR"] = df["AUR"].astype(float)
    df["AUR_ci"] = df["AUR_ci"].astype(float)
    df["Cohens $\\kappa$_ci"] = df["Cohens $\\kappa$_ci"].astype(float)
    df["MAE_ci"] = df["MAE_ci"].astype(float)

    df = df.rename(
        columns={
            "CohenKappa": "Cohen $\\kappa$",
            "MeanAverageError": "MAE",
            "AreaUnderReceiver": "AUC",
            "model": "Model",
            "components": "Components",
            "missing_qrels_load": "Missing",
        },
    )

    # Format metrics with gray ± CI next to absolute values
    def fmt_ci(val, ci):
        try:
            if pd.isna(val):
                return ""
            if pd.isna(ci):
                return f"{val:.2f}"
            return f"{val:.2f} \\textcolor{{gray}}{{$\\pm$ {ci:.2f}}}"
        except Exception:
            return str(val)

    df["AUR"] = df.apply(lambda r: fmt_ci(r["AUR"], r["AUR_ci"]), axis=1)
    df["Cohens $\\kappa$"] = df.apply(
        lambda r: fmt_ci(r["Cohens $\\kappa$"], r["Cohens $\\kappa$_ci"]), axis=1
    )
    df["MAE"] = df.apply(lambda r: fmt_ci(r["MAE"], r["MAE_ci"]), axis=1)

    # Drop CI helper columns from final table
    df = df.drop(
        columns=[
            "AUR_ci",
            "Cohens $\kappa$_ci",
            "MAE_ci",
        ],
    )

    df = df.sort_values(["Components", "Model"]).set_index(["Components", "Model"])

    # Only format numeric columns; metrics are now pre-formatted strings
    # formatters = {
    #     "Missing": lambda x: f"{x:.0f}",
    # }

    table = df.to_latex(
        "publication/paper/tables/agreement_robust_reference.tex",
        index=True,
        escape=False,
        # formatters=formatters,
        column_format="llcccc",
    )
    print(table)


if __name__ == "__main__":
    dataset = "robust"
    input_ = "qrels-topics-masked"
    main(dataset, input_)
