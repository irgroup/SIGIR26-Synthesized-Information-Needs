import datetime
import json
import os
from pathlib import Path
from src.data import DATA_DIR_PROCESSED, get_DNA_qrels


def parse_original_dna_qrels():
    qrels = get_DNA_qrels()

    output = DATA_DIR_PROCESSED / "qrels"

    # Save output
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.mkdir(Path(output) / timestamp)
    with open(Path(output) / timestamp / "metadata.json", "w") as f:
        json.dump({
            "date": timestamp,
            "model": "gpt-4.1",
            "data": "robust",
            "prompt": "-DNA-zero-shot",
            "k": None,
            "s": True,
            "topics": {
                "date": None,
                "model": "trec assessors",
                "data": "robust",
                "prompt": None,
                "k": None,
                "s": True,
                "task": "topics",
            },
            "task": "qrels",
        }, f)

    qrels.to_csv(output / timestamp / "qrels.csv.gz", index=False, header=False,
                 sep=" ", compression="gzip")


if __name__ == "__main__":
    parse_original_dna_qrels()
