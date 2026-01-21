#!/usr/bin/env python3
from glob import glob
from tempfile import TemporaryDirectory
from pathlib import Path
from tqdm import tqdm
from subprocess import check_output
import pandas as pd
from statistics import mean
from src.io import read_metadata

def allowed_qrels(dataset, topic_type):
    metadata = read_metadata(f'../data/interim/{dataset}/qrels-{topic_type}')
    df = metadata[(metadata['topics_nqueries'] == 1.0) & (metadata["topics_prompt"].isin(["topic-query", "topic-query-contrastive"]))]
    return set(df["name"].unique())
    

def all_qrels(dataset, topic_type):
    for qrel_file in tqdm(glob(f'../data/interim/{dataset}/qrels-{topic_type}/**/qrels.csv.gz')):
        raise ValueError(qrel_file.parent.parent.name)
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

line = "  & & "

ret = {}

#for ds in ["dl19", "dl20", "robust"]:
for ds in ["dl19", "dl20"]:
    ret[ds] = {}
    for qrel_type in ["reference", "topics-generated-title-description", "topics-generated-full", "topics-generated-title-narrative"]:
        rels = []
        print(len(list(all_qrels(ds, qrel_type))))
        for rel in all_qrels(ds, qrel_type):
            rels.append(rel)
        ret[ds][qrel_type] = str(int((mean(rels) * 10000))/100)

for k, v in [("reference", "\\cmark & \\xmark & \\xmark"), ("topics-generated-title-description", "\\cmark & \\cmark & \\xmark"), ("topics-generated-title-narrative", "\\cmark & \\xmark & \\cmark"), ("topics-generated-full", "\\cmark & \\cmark & \\cmark")]:
    l = [v, "---", "---", ret["dl19"][k], ret["dl20"][k]]
    print(" & ".join(l) + "\\\\")

