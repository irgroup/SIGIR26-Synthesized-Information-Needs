import datetime
import json
import os
from pathlib import Path
from src.data import DATA_DIR_INTERIM, get_dataset


def parse_original_dna_qrels():
    robust = get_dataset("robust")
    qrels = robust._sample_test_qrels()
    output = DATA_DIR_INTERIM / "robust-reference"

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
            "topics": {
                "date": timestamp,
                "model": "Human",
                "data": "robust",
                "prompt": None,
                "k": None,
                "nqueries": None,
                "ndocspos": None,
                "ndocsneg": None,
                "output": output.as_posix(),
                "task": "topics"
            },
            "output": output.as_posix(),
            "task": "qrels",
        }, f)

    qrels.to_csv(output / timestamp / "qrels.csv.gz", index=False, header=False,
                 sep=" ", compression="gzip")


if __name__ == "__main__":
    parse_original_dna_qrels()
