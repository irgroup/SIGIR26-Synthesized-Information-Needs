from src.data import DATA_DIR_PROCESSED, get_DNA_qrels


def parse_original_dna_qrels():
    qrels = get_DNA_qrels()

    # save to processed data directory
    output_file = DATA_DIR_PROCESSED / "exp2" /\
        "qrels-robust_gpt-4.1_-DNA-zero-shot_topics-trec_s.csv.gz"
    qrels.to_csv(output_file, index=False, header=False,
                 sep=" ", compression="gzip")


if __name__ == "__main__":
    parse_original_dna_qrels()
