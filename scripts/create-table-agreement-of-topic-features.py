#!/usr/bin/env python3
from glob import glob
from tempfile import TemporaryDirectory
from pathlib import Path
from tqdm import tqdm
from subprocess import check_output
import pandas as pd
from statistics import mean

def all_qrels(dataset):
    for qrel_file in tqdm(glob(f'../data/interim/{dataset}/qrels-topics-generated/**/qrels.csv.gz')):
        target_file = Path(f'../data/interim/{dataset}/qrels-analyzed') / Path(qrel_file).parent.name / "qrels-analyzed.jsonl.gz"
        if not target_file.is_file():
            target_file.parent.mkdir(parents=True, exist_ok=True)
            check_output(["reliability-tests", "analyze", "--qrels", qrel_file, "--output", target_file])
        ret = pd.read_json(target_file, lines=True)
        ret = ret[ret["Measure"] == "qrels-relevant-mean"]
        assert len(ret) == 1
        ret = ret.iloc[0].to_dict()
        ret = [v for k, v in ret.items() if k not in ("Measure", "Aspect")]
        assert len(ret) == 1
        yield ret[0]

line = "\\cmark & \\xmark & \\xmark  & & "

for ds in ["dl19", "dl20", "robust"]:
    rels = []
    for rel in all_qrels(ds):
        rels.append(rel)


    line += " & " + str(int((mean(rels) * 10000))/100) + "\\,\\%"

print(line)

