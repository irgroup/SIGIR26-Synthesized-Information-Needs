import pandas as pd

from src.config import MODEL_SORTER, PROMPT_SORTER
from src.data import DATA_DIR_PROCESSED

DATASETS = {
    "Robust": {
        "binary": "label-alignment-robust-qrels-topics-generated-binary.tsv",
        "graded": "label-alignment-robust-qrels-topics-generated.tsv",
    },
    "DL19": {
        "binary": "label-alignment-dl19-qrels-topics-generated-full-binary.tsv",
        "graded": "label-alignment-dl19-qrels-topics-generated-full.tsv",
    },
    "DL20": {
        "binary": "label-alignment-dl20-qrels-topics-generated-full-binary.tsv",
        "graded": "label-alignment-dl20-qrels-topics-generated-full.tsv",
    },
}

RENAME_MAP = {
    "topics_nqueries": "$q$",
    "topics_ndocspos": "$d^+$",
    "topics_ndocsneg": "$d^-$",
    "model": "Model",
    "topics_prompt": "Prompt",
    "missing_qrels": "X",
    "label_dist(0)": "0",
    "label_dist(1)": "1",
    "label_dist(2)": "2",
    "label_dist(3)": "3",
    "Cohens $\\kappa$": "$\\kappa$",
    "Cohens $\\kappa$_ci": "$\\kappa$_ci",
}


def _prepare_dataset(
    file_name: str, dataset_label: str, grade_type: str, model_filter: str | None
) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR_PROCESSED / file_name, sep="\t")
    df = df.drop_duplicates()
    df = df.pivot(index="name", columns="measure", values="value").reset_index()

    # fix model name
    df["topics_model"] = df["topics_model"].str.replace(
        "GPT-OSS-120B-O", "GPT-OSS-120B"
    )

    # use only rows where the topic and judgment are done by the same LLM
    df = df[(df["model"] == df["topics_model"]) | (df["topics_prompt"] == "human")]

    df["model"] = pd.Categorical(df["model"], MODEL_SORTER)
    df["topics_prompt"] = pd.Categorical(df["topics_prompt"], PROMPT_SORTER)

    df = df.rename(columns=RENAME_MAP)

    if model_filter:
        df = df[df["Model"] == model_filter]

    # Determine which columns to include based on grade_type
    base_cols = ["Prompt", "$q$", "$d^+$", "$d^-$", "Model"]
    metric_base = ["$\\kappa$", "$\\kappa$_ci", "MAE", "MAE_ci"]

    if grade_type == "binary":
        label_cols = ["X", "0", "1"]
    elif grade_type == "graded":
        # Check which label columns exist
        label_cols = []
        for col in ["0", "1", "2", "3"]:
            if col in df.columns:
                label_cols.append(col)
        # Add X to show missing qrels count
        if "X" in df.columns:
            label_cols.append("X")
    else:
        label_cols = []

    available_cols = base_cols + metric_base + label_cols
    df = df[[col for col in available_cols if col in df.columns]]

    # Convert count columns to integers
    int_cols = ["$q$", "$d^+$", "$d^-$"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Round numeric columns
    numeric_cols = ["$\\kappa$", "MAE", "$\\kappa$_ci", "MAE_ci"] + [
        c for c in label_cols if c != "X"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    # Format label distribution columns without trailing zeros
    for col in label_cols:
        if col in df.columns and col != "X":
            df[col] = df[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else v)

    # Combine measure and CI columns into "value±ci" format
    measures = [
        ("$\\kappa$", "$\\kappa$_ci"),
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
    if "X" in label_cols:
        # Put X first, then metrics, then other labels
        ordered_labels = [c for c in label_cols if c != "X"]
        final_metric_cols = ["X", "$\\kappa$", "MAE"]
        final_metric_cols = final_metric_cols + ordered_labels
    else:
        final_metric_cols = ["$\\kappa$", "MAE"] + label_cols
    df = df[base_cols + final_metric_cols]

    # If model is filtered, don't need Model in the index
    if model_filter:
        index_cols = ["Prompt", "$q$", "$d^+$", "$d^-$"]
    else:
        index_cols = ["Prompt", "$q$", "$d^+$", "$d^-$", "Model"]

    df = df.set_index(index_cols)

    metric_cols = [col for col in final_metric_cols if col in df.columns]
    df = df[metric_cols]

    # Create multi-level column index: (Dataset, Grade Type, Metric)
    df.columns = pd.MultiIndex.from_product(
        [[dataset_label], [grade_type], metric_cols]
    )
    return df


def main(model: str | None = None, include_binary: bool = True):
    tables = []
    for dataset_label, files in DATASETS.items():
        for grade_type, file_name in files.items():
            if not include_binary and grade_type == "binary":
                continue
            table = _prepare_dataset(
                file_name, dataset_label, grade_type, model_filter=model
            )
            tables.append(table)

    combined = pd.concat(tables, axis=1, join="outer").sort_index()
    combined = combined.fillna("-")

    print(combined)

    # Calculate column format: 3 or 4 index columns + metrics per dataset/type
    num_index_cols = 4 if model else 5  # Prompt, q, d+, d- (+ Model if not filtered)

    column_format = "l" * num_index_cols + "c" * len(combined.columns)

    suffix = "-no-binary" if not include_binary else ""
    filename = (
        f"publication/paper/tables/agreement-label-{model}{suffix}.tex"
        if model
        else f"publication/paper/tables/agreement-label-all-models{suffix}.tex"
    )
    latex_output = combined.to_latex(
        # filename,
        index=True,
        escape=False,
        column_format=column_format,
        multicolumn=True,
        multicolumn_format="c",
    )


if __name__ == "__main__":
    # main(model=None, include_binary=True)
    main(model="GPT-OSS-120B", include_binary=False)
    # main(include_binary=True)
