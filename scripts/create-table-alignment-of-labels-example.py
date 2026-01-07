import pandas as pd

from src.config import MODEL_SORTER, PROMPT_SORTER
from src.data import DATA_DIR_PROCESSED

DATASETS = {
    "Robust": "alignment-robust-qrels-topics-generated.tsv",
    "DL19": "alignment-dl19-qrels-topics-generated.tsv",
    "DL20": "alignment-dl20-qrels-topics-generated.tsv",
}

RENAME_MAP = {
    "topics_nqueries": "$q$",
    "topics_ndocspos": "$d^+$",
    "topics_ndocsneg": "$d^-$",
    "model": "Model",
    "topics_prompt": "Prompt",
    "missing_qrels_load": "X",
    "label_dist(0)": "0",
    "label_dist(1)": "1",
}


def _prepare_dataset(file_name: str, dataset_label: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR_PROCESSED / file_name, sep="\t")
    df = df.drop_duplicates()
    df = df.pivot(index="name", columns="measure", values="value").reset_index()
    df["topics_model"] = df["topics_model"].str.replace(
        "GPT-OSS-120B-O", "GPT-OSS-120B"
    )
    df = df[df["topics_model"] == "GPT-OSS-120B"]

    df = df[(df["model"] == df["topics_model"]) | (df["topics_prompt"] == "human")]
    df["model"] = pd.Categorical(df["model"], MODEL_SORTER)
    df["topics_prompt"] = pd.Categorical(df["topics_prompt"], PROMPT_SORTER)

    df = df.rename(columns=RENAME_MAP)

    df = df[
        [
            "Prompt",
            "$q$",
            "$d^+$",
            "$d^-$",
            "Model",
            "Cohens $\\kappa$",
            "Cohens $\\kappa$_ci",
            "MAE",
            "MAE_ci",
            "0",
            "1",
            "X",
        ]
    ]

    # Convert count columns to integers
    int_cols = ["$q$", "$d^+$", "$d^-$"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Round numeric columns - only if they are numeric
    numeric_cols = [
        "Cohens $\\kappa$",
        "MAE",
        "Cohens $\\kappa$_ci",
        "MAE_ci",
        "0",
        "1",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    # Format label distribution columns without trailing zeros
    for col in ["0", "1"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: f"{v:.2f}".rstrip("0").rstrip(".") if pd.notna(v) else v
            )

    # Combine measure and CI columns into "value±ci" format
    measures = [
        ("Cohens $\\kappa$", "Cohens $\\kappa$_ci"),
        ("MAE", "MAE_ci"),
    ]

    for measure_col, ci_col in measures:
        if measure_col in df.columns and ci_col in df.columns:
            df[measure_col] = df.apply(
                lambda row: f"{row[measure_col]:.2f}{{\\textcolor{{gray}}{{±{row[ci_col]:.2f}}}}}"
                if pd.notna(row[measure_col]) and pd.notna(row[ci_col])
                else str(row[measure_col]),
                axis=1,
            )

    # Keep only the main columns (drop separate CI columns)
    df = df[
        [
            "Prompt",
            "$q$",
            "$d^+$",
            "$d^-$",
            "Model",
            "Cohens $\\kappa$",
            "MAE",
            "0",
            "1",
            "X",
        ]
    ]

    df = df.sort_values(["Prompt", "$q$", "$d^+$", "$d^-$", "Model"])
    index_cols = ["Prompt", "$q$", "$d^+$", "$d^-$", "Model"]
    df = df.set_index(index_cols)

    metric_cols = ["Cohens $\\kappa$", "MAE", "0", "1", "X"]
    df = df[metric_cols]
    df.columns = pd.MultiIndex.from_product([[dataset_label], metric_cols])
    return df


def main():
    tables = [
        _prepare_dataset(file_name, label) for label, file_name in DATASETS.items()
    ]
    combined = pd.concat(tables, axis=1, join="outer").sort_index()
    combined = combined.fillna("-")

    # Five index columns + five metric columns per dataset
    column_format = "l" * 5 + "c" * (len(DATASETS) * 5)

    print(combined)
    latex_output = combined.to_latex(
        "publication/paper/tables/agreement_label_binary.tex",
        index=True,
        escape=False,
        column_format=column_format,
        multicolumn=True,
        multicolumn_format="c",
    )
    print(latex_output)


if __name__ == "__main__":
    main()
