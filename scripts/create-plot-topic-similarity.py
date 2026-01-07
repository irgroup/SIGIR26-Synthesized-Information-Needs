import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import DATA_DIR_PROCESSED


def main():
    # Load the dataset
    df = pd.read_csv(DATA_DIR_PROCESSED / "2similarity-robust-topics.tsv", sep="\t")
    df = df.drop_duplicates()

    # Filter for the three similarity measures
    similarity_measures = ["CosineSimilarity", "JaccardIndex", "RelativeLength"]
    pattern = "|".join([f"^{m}" for m in similarity_measures])
    df_sim = df[df["measure"].str.contains(pattern, regex=True, na=False)].copy()

    # Extract component (title, description, narrative) from measure column
    df_sim["component"] = df_sim["measure"].str.extract(r"\((.*?)\)")
    df_sim["metric"] = df_sim["measure"].str.extract(r"^([^(]+)")

    # Get metadata for each name (timestamp)
    metadata_measures = ["nqueries", "model", "prompt", "ndocspos", "ndocsneg"]
    metadata_dfs = []
    for measure in metadata_measures:
        temp_df = df[df["measure"] == measure][["name", "value"]].copy()
        temp_df.columns = ["name", measure]
        metadata_dfs.append(temp_df)

    # Merge all metadata
    metadata = metadata_dfs[0]
    for meta_df in metadata_dfs[1:]:
        metadata = metadata.merge(meta_df, on="name", how="outer")

    # Merge with similarity data
    df_sim = df_sim.merge(metadata, on="name", how="left")

    # Rename value column to similarity and convert to numeric
    df_sim = df_sim.rename(columns={"value": "similarity"})
    df_sim["similarity"] = pd.to_numeric(df_sim["similarity"], errors="coerce")

    # Convert nqueries to numeric
    df_sim["nqueries"] = pd.to_numeric(df_sim["nqueries"], errors="coerce")
    df_sim["ndocspos"] = pd.to_numeric(df_sim["ndocspos"], errors="coerce")
    df_sim["ndocsneg"] = pd.to_numeric(df_sim["ndocsneg"], errors="coerce")

    # Drop rows with missing values
    df_sim = df_sim.dropna(subset=["similarity", "nqueries", "model", "prompt"])

    # Filter for prompts that start with "topic-query"
    df_sim = df_sim[df_sim["prompt"].str.startswith("topic-query")]

    # Remove Llama3.3-70B from evaluation
    df_sim = df_sim[df_sim["model"] != "Llama3.3-70B"]

    # For topic-query-contrastive, only keep results where nqueries == ndocspos == ndocsneg
    contrastive_mask = (df_sim["prompt"] == "topic-query-contrastive") & (
        (df_sim["nqueries"] != df_sim["ndocspos"])
        | (df_sim["nqueries"] != df_sim["ndocsneg"])
    )
    df_sim = df_sim[~contrastive_mask]

    # Sort components in desired order
    df_sim["component"] = pd.Categorical(
        df_sim["component"],
        categories=["title", "description", "narrative"],
        ordered=True,
    )

    # Sort metrics in desired order
    df_sim["metric"] = pd.Categorical(
        df_sim["metric"],
        categories=["CosineSimilarity", "JaccardIndex", "RelativeLength"],
        ordered=True,
    )

    # Filter for maximum 5 queries
    df_sim = df_sim[df_sim["nqueries"] <= 5]

    # Create the relplot with facets: columns per component, rows per measure
    g = sns.relplot(
        data=df_sim,
        x="nqueries",
        y="similarity",
        hue="model",
        style="prompt",
        col="component",
        row="metric",
        kind="line",
        # markers=True,
        dashes=True,
        height=3,
        aspect=1.2,
        facet_kws={"sharex": True, "sharey": False},
        errorbar=None,  # Disable error bars since we aggregated
    )

    # Customize the plot
    g.set_axis_labels("Number of Queries", "Similarity Score")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.fig.suptitle("Topic Similarity Measures by Component", y=1.02, fontsize=14)

    # Set x-axis to show each query number
    for ax in g.axes.flat:
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(0.5, 5.5)

    output_path = "publication/paper/figures/topic-similarity.pdf"
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
