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


def all_logo_tests(dataset):
    for qrel_file in tqdm(glob(f'../data/interim/{dataset}/qrels-topics-generated/**/qrels.csv.gz')):
        target_file = Path(f'../data/interim/{dataset}/qrels-analyzed') / Path(qrel_file).parent.name / "top-10-logo.jsonl.gz"
        if not target_file.is_file():
            write_job_yaml(dataset, Path(qrel_file).parent.name)

def write_job_yaml(dataset, qrels_file):
    identifier = f"{dataset}-{qrels_file}".replace(":", "-").replace("_", "-")
    if dataset == "dl19":
        experiment_config = "trec-28"
    else:
        raise ValueError("foo")
    
    yaml = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: gen-{identifier}
  namespace: kibi9872
  labels:
    jobgroup: reliability-jobs
spec:
  parallelism: 1
  template:
    metadata:
      name: reliability-jobs
      labels:
        jobgroup: reliability-jobs
    spec:
      containers:
      - name: c
        image: mam10eks/repro-eval:prod
        imagePullPolicy: Always
        command: ["reliability-tests", "run-test", "--tests", "top-10-logo", "--input", "/data/experiments/{experiment_config}", "--qrels", "/outputs/{dataset}/qrels-topics-generated/{qrels_file}/qrels.csv.gz", "--output", "/outputs/{dataset}/qrels-analyzed/{qrels_file}/top-10-logo"]
        volumeMounts:
          - mountPath: "/mnt/ceph/storage/data-in-progress/data-research/web-search/web-search-trec/trec-system-runs/"
            name: run-dir
            readOnly: true
          - mountPath: "/data"
            name: data-dir
            readOnly: true
        resources:
          requests:
            memory: 10Gi
            cpu: 1
          limits:
            memory: 15Gi
            cpu: 1
      volumes:
        - name: run-dir
          hostPath:
            path: /mnt/ceph/storage/data-in-progress/data-research/web-search/web-search-trec/trec-system-runs/
            type: Directory
        - name: data-dir
          hostPath:
            path: /mnt/ceph/storage/data-tmp/current/kibi9872/conf26-reliability-analysis/experiments/data
            type: Directory
        - name : outputs
          hostPath:
            path: /mnt/ceph/storage/data-tmp/current/kibi9872/conf26-generating-topics/data/interim
            type: Directory
      restartPolicy: Never
"""
    with open(f"jobs/{identifier}.yml", "w") as f:
        f.write(yaml)

for ds in ["dl19"]:
    rels = []
    all_logo_tests(ds)

line = "\\cmark & \\xmark & \\xmark  & & "

for ds in ["dl19", "dl20", "robust"]:
    rels = []
    for rel in all_qrels(ds):
        rels.append(rel)


    line += " & " + str(int((mean(rels) * 10000))/100) + "\\,\\%"

print(line)

