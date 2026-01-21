#!/usr/bin/env python3
from glob import glob
from tqdm import tqdm
import json
import gzip
from pathlib import Path
from statistics import mean

def all_logo_tests(dataset):
#    for qrel_file in tqdm(glob(f'../data/interim/{dataset}/qrels-topics-generated-title/**/qrels.csv.gz'), dataset):
#        target_file = Path(f'../data/interim/{dataset}/qrels-analyzed') / Path(qrel_file).parent.name / "top-10-logo" / "reliability-test-results.json.gz"
#        if not target_file.is_file():
#            write_job_yaml(dataset, Path(qrel_file).parent.name, "qrels-topics-generated-title")
#
#    for qrel_file in tqdm(glob(f'../data/interim/{dataset}/qrels-topics-generated-full/**/qrels.csv.gz'), dataset):
#        target_file = Path(f'../data/interim/{dataset}/qrels-analyzed') / Path(qrel_file).parent.name / "top-10-logo" / "reliability-test-results.json.gz"
#        if not target_file.is_file():
#            write_job_yaml(dataset, Path(qrel_file).parent.name, "qrels-topics-generated-full")
#
    for qrel_file in tqdm(glob(f'../data/interim/{dataset}/qrels-topics-generated-title-description/**/qrels.csv.gz'), dataset):
        target_file = Path(f'../data/interim/{dataset}/qrels-analyzed') / Path(qrel_file).parent.name / "top-10-logo" / "reliability-test-results.json.gz"
        if not target_file.is_file():
            write_job_yaml(dataset, Path(qrel_file).parent.name, "qrels-topics-generated-title-description")

    for qrel_file in tqdm(glob(f'../data/interim/{dataset}/qrels-topics-generated-title-narrative/**/qrels.csv.gz'), dataset):
        target_file = Path(f'../data/interim/{dataset}/qrels-analyzed') / Path(qrel_file).parent.name / "top-10-logo" / "reliability-test-results.json.gz"
        if not target_file.is_file():
            write_job_yaml(dataset, Path(qrel_file).parent.name, "qrels-topics-generated-title-narrative")


def write_job_yaml(dataset, qrels_file, directory):
    identifier = f"{dataset}-{qrels_file}".replace(":", "-").replace("_", "-")
    if dataset == "dl19":
        experiment_config = "trec-28"
    elif dataset == "dl20":
        experiment_config = "trec-29"
    else:
        raise ValueError("foo")
    qrel_mapping = '{\\"999\\": \\"0\\", \\"6\\": \\"0\\"}'
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
        command: ["reliability-tests", "run-test", "--tests", "top-10-logo", "--map-qrels", "{qrel_mapping}", "--input", "/data/experiments/{experiment_config}", "--qrels", "/outputs/{dataset}/{directory}/{qrels_file}/qrels.csv.gz", "--output", "/outputs/{dataset}/qrels-analyzed/{qrels_file}/top-10-logo"]
        volumeMounts:
          - mountPath: "/mnt/ceph/storage/data-in-progress/data-research/web-search/web-search-trec/trec-system-runs/"
            name: run-dir
            readOnly: true
          - mountPath: "/data"
            name: data-dir
            readOnly: true
          - mountPath: "/outputs"
            name: outputs
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

def run_all_logo_tests():
    for ds in ["dl19", "dl20"]:
        rels = []
        all_logo_tests(ds)


def allowed_qrels(dataset, topic_type):
    if "reference" == topic_type:
        return None
    
    ret = json.load(open('../data/interim/qrels_metadata.json'))
    return set(ret["qrels_topic_features"][dataset][f"qrels-{topic_type}"])        

def print_results():
    ret = {}
    for dataset in ["dl19", "dl20"]:
        ret[dataset] = {}
        qrel_files = {"reference": [], "topics-generated-title-description": [], "topics-generated-full": [], "topics-generated-title-narrative": []}

        for t in qrel_files.keys():
            allow_list = allowed_qrels(dataset, t)
            for qrel_file in glob(f'../data/interim/{dataset}/qrels-{t}/**/qrels.csv.gz'):
                if allow_list is not None:
                    if Path(qrel_file).parent.name not in allow_list:
                        continue
                metadata_file = Path(qrel_file).parent / "metadata.json"
                metadata = json.loads(metadata_file.read_text())

                #if metadata["topics"]["prompt"] in ALLOWED_TOPIC_FORMATS:
                qrel_files[t].append(Path(qrel_file).parent.name)


        print({k: len(v) for k, v in qrel_files.items()})

        def form(corr, measure):
            return f"{mean(corr_to_measure_to_vals[corr][measure]):.3f}"

        skipped = 0

        for t in qrel_files.keys():
            corr_to_measure_to_vals = {"spearman": {"nDCG@10": [], "nDCG@20": [], "nDCG": []}, "tauap_b": {"nDCG@10": [], "nDCG@20": [], "nDCG": []}}
            for corr in corr_to_measure_to_vals.keys():
                for qrel_file in qrel_files[t]:
                    target_file = Path(f"../data/interim/{dataset}/qrels-analyzed/{qrel_file}/top-10-logo/reliability-test-results.json.gz")
                    if not target_file.exists():
                        skipped += 1
                        print(target_file)
                        continue
                    with gzip.open(target_file, "rt") as f:
                        i = json.loads(f.read())
                        measures = {"nDCG@10": [], "nDCG@20": [], "nDCG": []}
                        for r in i["system_ranking_evaluation"]:
                            for m in measures.keys():
                                measures[m].append(r[m][corr])
                        for m in measures.keys():
                            corr_to_measure_to_vals[corr][m].append(mean(measures[m]))
 



            ret[dataset][t] = {
                "nDCG@10": {
                    "spearman": form("spearman", "nDCG@10"),
                    "tauap_b": form("tauap_b", "nDCG@10"),
                },
                "nDCG@20": {
                    "spearman": form("spearman", "nDCG@20"),
                    "tauap_b": form("tauap_b", "nDCG@20"),
                },
                "nDCG": {
                    "spearman": form("spearman", "nDCG"),
                    "tauap_b": form("tauap_b", "nDCG"),
                }
            }

    print("skipped", skipped)

    for field, prefix in [("reference", "\\cmark & \\xmark & \\xmark"), ("topics-generated-title-description", "\\cmark & \\cmark & \\xmark"), ("topics-generated-title-narrative", "\\cmark & \\xmark & \\cmark"), ("topics-generated-full", "\\cmark & \\cmark & \\cmark")]:
        line = []
        for corr in ["spearman", "tauap_b"]:
            line += [ret["dl19"][field]["nDCG@10"][corr], ret["dl20"][field]["nDCG@10"][corr]]
            line += [ret["dl19"][field]["nDCG@20"][corr], ret["dl20"][field]["nDCG@20"][corr]]
            line += [ret["dl19"][field]["nDCG"][corr], ret["dl20"][field]["nDCG"][corr]]
        print(prefix + " & " + " & ".join(line) + "\\\\")


if __name__ == '__main__':
    #run_all_logo_tests()
    print_results()
