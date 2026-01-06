#!/usr/bin/env python3
from glob import glob
from tempfile import TemporaryDirectory
from pathlib import Path
from tqdm import tqdm
from subprocess import check_output

def all_qrels(dataset):
    for qrel_file in tqdm(glob(f'../data/interim/{dataset}/qrels-topics-generated/**/qrels.csv.gz')):
        target_file = Path(f'../data/interim/{dataset}/qrels-analyzed') / Path(qrel_file).parent.name / "qrels-analyzed.jsonl.gz"
        if target_file.is_file():
            continue
        target_file.parent.mkdir(parents=True, exist_ok=True)
        check_output(["reliability-tests", "analyze", "--qrels", qrel_file, "--output", target_file])


for ds in ["dl19", "dl20", "robust"]:
    all_qrels(ds)


