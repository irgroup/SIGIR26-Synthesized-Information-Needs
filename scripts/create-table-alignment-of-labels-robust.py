import pandas as pd

from src.config import COMPONENTS_SORTER, MODEL_SORTER
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
    # Load binary results
    df_binary = pd.read_csv(
        DATA_DIR_PROCESSED / f"alignment-{dataset}-{input_}-binary.tsv", sep="\t"
    )
    df_binary = df_binary[df_binary["name"] != "disks45/nocr/trec-robust-2004"]

    df_binary = df_binary.pivot(index="name", columns="measure", values="value")
    df_binary.columns.name = None
    df_binary = df_binary.reset_index()

    # Load graded results
    df_graded = pd.read_csv(
        DATA_DIR_PROCESSED / f"alignment-{dataset}-{input_}-graded.tsv", sep="\t"
    )
    df_graded = df_graded[df_graded["name"] != "disks45/nocr/trec-robust-2004"]

    df_graded = df_graded.pivot(index="name", columns="measure", values="value")
    df_graded.columns.name = None
    df_graded = df_graded.reset_index()

    # Process binary dataframe
    df_binary["components"] = df_binary["prompt"].apply(prompt_to_components)
    df_binary["components"] = pd.Categorical(df_binary["components"], COMPONENTS_SORTER)
    df_binary["model"] = pd.Categorical(df_binary["model"], MODEL_SORTER)

    df_binary = df_binary[
        [
            "model",
            "components",
            "Cohens $\\kappa$",
            "MAE",
            "missing_qrels_load",
            "Cohens $\\kappa$_ci",
            "MAE_ci",
            "label_dist(0)",
            "label_dist(1)",
        ]
    ]
    # convert to floats
    df_binary["Cohens $\\kappa$"] = df_binary["Cohens $\\kappa$"].astype(float)
    df_binary["MAE"] = df_binary["MAE"].astype(float)
    df_binary["Cohens $\\kappa$_ci"] = df_binary["Cohens $\\kappa$_ci"].astype(float)
    df_binary["MAE_ci"] = df_binary["MAE_ci"].astype(float)
    df_binary["label_dist(0)"] = df_binary["label_dist(0)"].astype(float).round(2)
    df_binary["label_dist(1)"] = df_binary["label_dist(1)"].astype(float).round(2)

    # Process graded dataframe
    df_graded["components"] = df_graded["prompt"].apply(prompt_to_components)
    df_graded["components"] = pd.Categorical(df_graded["components"], COMPONENTS_SORTER)
    df_graded["model"] = pd.Categorical(df_graded["model"], MODEL_SORTER)

    df_graded = df_graded[
        [
            "model",
            "components",
            "Cohens $\\kappa$",
            "MAE",
            "Cohens $\\kappa$_ci",
            "MAE_ci",
            "label_dist(0)",
            "label_dist(1)",
            "label_dist(2)",
        ]
    ]
    # convert to floats
    df_graded["Cohens $\\kappa$"] = df_graded["Cohens $\\kappa$"].astype(float)
    df_graded["MAE"] = df_graded["MAE"].astype(float)
    df_graded["Cohens $\\kappa$_ci"] = df_graded["Cohens $\\kappa$_ci"].astype(float)
    df_graded["MAE_ci"] = df_graded["MAE_ci"].astype(float)
    df_graded["label_dist(0)"] = df_graded["label_dist(0)"].astype(float).round(2)
    df_graded["label_dist(1)"] = df_graded["label_dist(1)"].astype(float).round(2)
    df_graded["label_dist(2)"] = df_graded["label_dist(2)"].astype(float).round(2)

    # Merge binary and graded results
    # Rename columns to distinguish between binary and graded
    df_binary = df_binary.rename(
        columns={
            "Cohens $\\kappa$": "Cohens $\\kappa$ (bin)",
            "MAE": "MAE (bin)",
            "Cohens $\\kappa$_ci": "Cohens $\\kappa$_ci (bin)",
            "MAE_ci": "MAE_ci (bin)",
            "label_dist(0)": "0 (bin)",
            "label_dist(1)": "1 (bin)",
        },
    )

    df_graded = df_graded.rename(
        columns={
            "Cohens $\\kappa$": "Cohens $\\kappa$ (grad)",
            "MAE": "MAE (grad)",
            "Cohens $\\kappa$_ci": "Cohens $\\kappa$_ci (grad)",
            "MAE_ci": "MAE_ci (grad)",
            "label_dist(0)": "0 (grad)",
            "label_dist(1)": "1 (grad)",
            "label_dist(2)": "2 (grad)",
        },
    )

    # Merge on name, model, components
    df = df_binary.merge(
        df_graded[
            [
                "model",
                "components",
                "Cohens $\\kappa$ (grad)",
                "MAE (grad)",
                "Cohens $\\kappa$_ci (grad)",
                "MAE_ci (grad)",
                "0 (grad)",
                "1 (grad)",
                "2 (grad)",
            ]
        ],
        on=["model", "components"],
        how="left",
    )

    df = df.rename(
        columns={
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

    # Format binary metrics
    df["Cohens $\\kappa$ (bin)"] = df.apply(
        lambda r: fmt_ci(r["Cohens $\\kappa$ (bin)"], r["Cohens $\\kappa$_ci (bin)"]),
        axis=1,
    )
    df["MAE (bin)"] = df.apply(
        lambda r: fmt_ci(r["MAE (bin)"], r["MAE_ci (bin)"]), axis=1
    )

    # Format graded metrics
    df["Cohens $\\kappa$ (grad)"] = df.apply(
        lambda r: fmt_ci(r["Cohens $\\kappa$ (grad)"], r["Cohens $\\kappa$_ci (grad)"]),
        axis=1,
    )
    df["MAE (grad)"] = df.apply(
        lambda r: fmt_ci(r["MAE (grad)"], r["MAE_ci (grad)"]), axis=1
    )

    # Drop CI helper columns from final table
    df = df.drop(
        columns=[
            "Cohens $\kappa$_ci (bin)",
            "MAE_ci (bin)",
            "Cohens $\kappa$_ci (grad)",
            "MAE_ci (grad)",
        ],
    )

    df = df.sort_values(["Components", "Model"]).set_index(["Components", "Model"])

    # Create MultiIndex for columns
    # Reorder and rename columns for multiindex
    column_order = [
        ("Missing", ""),
        ("Binary", "Cohens $\\kappa$"),
        ("Binary", "MAE"),
        ("Binary", "0"),
        ("Binary", "1"),
        ("Graded", "Cohens $\\kappa$"),
        ("Graded", "MAE"),
        ("Graded", "0"),
        ("Graded", "1"),
        ("Graded", "2"),
    ]

    # Rename columns to match multiindex structure
    df = df.rename(
        columns={
            "Cohens $\\kappa$ (bin)": ("Binary", "Cohens $\\kappa$"),
            "MAE (bin)": ("Binary", "MAE"),
            "0 (bin)": ("Binary", "0"),
            "1 (bin)": ("Binary", "1"),
            "Cohens $\\kappa$ (grad)": ("Graded", "Cohens $\\kappa$"),
            "MAE (grad)": ("Graded", "MAE"),
            "0 (grad)": ("Graded", "0"),
            "1 (grad)": ("Graded", "1"),
            "2 (grad)": ("Graded", "2"),
            "Missing": ("Missing", ""),
        }
    )

    # Create MultiIndex from tuples
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    # Reorder columns
    df = df[[col for col in column_order if col in df.columns]]

    # Only format numeric columns; metrics are now pre-formatted strings
    formatters = {
        ("Binary", "0"): lambda x: f"{x:.2f}",
        ("Binary", "1"): lambda x: f"{x:.2f}",
        ("Graded", "0"): lambda x: f"{x:.2f}",
        ("Graded", "1"): lambda x: f"{x:.2f}",
        ("Graded", "2"): lambda x: f"{x:.2f}",
    }
    print(df)

    table = df.to_latex(
        "publication/paper/tables/alignment-labels-robust-masked.tex",
        index=True,
        escape=False,
        formatters=formatters,
        column_format="lccccccccc",
        multicolumn_format="c",
    )


if __name__ == "__main__":
    dataset = "robust"
    input_ = "qrels-topics-masked"
    main(dataset, input_)
