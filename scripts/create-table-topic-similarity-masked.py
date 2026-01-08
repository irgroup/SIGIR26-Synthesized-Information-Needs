import re
from pathlib import Path

import pandas as pd

from src.data import DATA_DIR_PROCESSED
from src.config import MODEL_SORTER, PROMPT_SORTER

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

def _ordered_multiindex(df, field_order=None, measure_order=None):
    pattern = re.compile(r"(?P<measure>.+)\((?P<field>.+)\)")
    tuples = []
    orig_cols = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            measure = m.group("measure")
            field = m.group("field")
            tuples.append((field, measure))
            orig_cols.append(c)

    if not tuples:
        return [], None

    if field_order is None:
        field_order = ["Title", "Description", "Narrative"]

    if measure_order is None:
        seen = []
        for _, meas in tuples:
            if meas not in seen:
                seen.append(meas)
        measure_order = seen

    ordered_tuples = []
    ordered_cols = []
    used = set()
    for f in field_order:
        for m in measure_order:
            for i, (ff, mm) in enumerate(tuples):
                if i in used:
                    continue
                if ff == f and mm == m:
                    ordered_tuples.append((f, m))
                    ordered_cols.append(orig_cols[i])
                    used.add(i)
                    break

    mi = pd.MultiIndex.from_tuples(ordered_tuples)
    return ordered_cols, mi


def main():
    # df = pd.read_csv(DATA_DIR_PROCESSED / "similarity-robust-topics.tsv", sep="\t")
    df = pd.read_csv(DATA_DIR_PROCESSED / "similarity-robust-topics-masked.tsv", sep="\t")
    df = df.drop_duplicates()
    df = df.pivot(index="name", columns="measure", values="value").reset_index()
    

    # Create mapping for column renaming
    rename_map = {
        "model": "Model",
        "prompt": "Fields",
        "CosineSimilarity(title)": "Cos(Title)",
        "CosineSimilarity(description)": "Cos(Description)",
        "CosineSimilarity(narrative)": "Cos(Narrative)",
        "JaccardIndex(title)": "Jacc.(Title)",
        "JaccardIndex(description)": "Jacc.(Description)",
        "JaccardIndex(narrative)": "Jacc.(Narrative)",
        "RelativeLength(title)": "Len(Title)",
        "RelativeLength(description)": "Len(Description)",
        "RelativeLength(narrative)": "Len(Narrative)",
    }

    df = df.rename(columns=rename_map)
    df["Fields"] = df["Fields"].apply(prompt_to_components)

    fixed_cols = [
        c for c in ["Fields", "Model"] if c in df.columns
    ]

    # detect measure columns like 'Cos(Title)' (after renaming)
    # and also track confidence intervals
    measure_cols = []
    ci_cols_map = {}  # maps original col name to its CI col

    for col in df.columns:
        if "(" in col and ")" in col and col not in fixed_cols:
            # Check if this is a base measure (not a CI variant)
            if not any(suffix in col for suffix in ["_ci", "_ci_lower", "_ci_upper"]):
                measure_cols.append(col)
                # Check for corresponding CI column
                ci_col = col + "_ci"
                if ci_col in df.columns:
                    ci_cols_map[col] = ci_col

    ordered_cols, mi = _ordered_multiindex(df[measure_cols])
    if not ordered_cols:
        # fallback: keep original selection if parsing failed
        print("No measure columns parsed; printing head instead.")
        print(df.head())
        return

    measure_df = df[ordered_cols].copy()
    measure_df = measure_df.apply(pd.to_numeric, errors="coerce")

    # Add confidence intervals in light grey
    for col in ordered_cols:
        # Find the original column name (before renaming) to look up CI
        orig_col = None
        for orig, renamed in rename_map.items():
            if renamed == col:
                orig_col = orig
                break

        ci_col_name = orig_col + "_ci" if orig_col else col + "_ci"

        if ci_col_name in df.columns:
            ci_values = pd.to_numeric(df[ci_col_name], errors="coerce")
            # Format main value with ci appended in light grey
            formatted = measure_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else ""
            )
            formatted = formatted.str.cat(
                ci_values.apply(
                    lambda x: f" {{\\color{{gray}} ±{x:.3f}}}" if pd.notna(x) else ""
                ),
                sep="",
            )
            measure_df[col] = formatted
        else:
            measure_df[col] = measure_df[col].round(2).astype(str)

    out_df = pd.concat(
        [df[fixed_cols].reset_index(drop=True), measure_df.reset_index(drop=True)],
        axis=1,
    )

    # Sort by Prompt (with q, d+, d-) and then by Model
    sort_cols = ["Fields", "Model"]
    sort_cols = [c for c in sort_cols if c in out_df.columns]
    
    out_df["Model"] = pd.Categorical(out_df["Model"], MODEL_SORTER)
    # out_df["Prompt"] = pd.Categorical(out_df["Prompt"], PROMPT_SORTER)
    out_df = out_df.sort_values(by=sort_cols).reset_index(drop=True)

    # Add missing_topics column at the end if it exists
    if "missing_topics" in df.columns:
        missing = pd.to_numeric(df["missing_topics"], errors="coerce").astype("Int64")
        out_df["Missing"] = missing


    tuples_all = [("", c) for c in fixed_cols] + list(mi)
    if "Missing" in out_df.columns:
        tuples_all.append(("", "Missing"))
    out_df.columns = pd.MultiIndex.from_tuples(tuples_all)

    tex = out_df.to_latex(index=False, multicolumn=True, escape=False)
    tex = tex.replace("nan", "-")
    tex = tex.replace("0.0", "-")
    print(tex)
    out_file = Path("../paper/tables/topic_masked_similarity.tex")
    # out_file.write_text(tex)
    print(f"Wrote LaTeX table to {out_file}")


if __name__ == "__main__":
    main()
