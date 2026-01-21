#!/usr/bin/env python3
from glob import glob
from tempfile import TemporaryDirectory
from pathlib import Path
from tqdm import tqdm
from subprocess import check_output
import pandas as pd
from statistics import mean
import json

def allowed_qrels(dataset, topic_type):
    if "reference" == topic_type:
        return None
    
    ret = json.load(open('../data/interim/qrels_metadata.json'))
    return set(ret["qrels_topic_features"][dataset][f"qrels-{topic_type}"])


def grouped_qrels(dataset, topic_type):
    candidates = glob(f'../data/interim/{dataset}/qrels-{topic_type}/**/qrels.csv.gz')

    if "reference" == topic_type:
        return [candidates]

    groups = json.load(open('../data/interim/qrels_metadata.json'))["qrels_annotator_agreement"][dataset][f"qrels-{topic_type}"]
    id_to_group = {}
    for group, keys in groups.items():
        for key in keys:
            id_to_group[key] = group
    ret_dict = {}

    for i in candidates:
        i = Path(i)
        name = i.parent.name
        if name not in id_to_group:
            continue
        group = id_to_group[name]
        if group not in ret_dict:
            ret_dict[group] = set()
        ret_dict[group].add(i)
    return [list(i) for i in ret_dict.values() if len(i) > 2]


def all_qrels(dataset, topic_type):
    allow_list = allowed_qrels(dataset, topic_type)
    for qrel_file in tqdm(glob(f'../data/interim/{dataset}/qrels-{topic_type}/**/qrels.csv.gz')):
        if allow_list is not None:
            if Path(qrel_file).parent.name not in allow_list:
                continue

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

ret = {}
ret_agr = {}

def do_agreement_calc(ds, qrel_type):
    groups = grouped_qrels(ds, qrel_type)
    from topic_gen.evaluate import Experiment
    from topic_gen.evaluate.measures_agreement_multiple import FleissKappa
    from ir_measures import read_trec_qrels
    from ir_measures.util import QrelsConverter
    import gzip

    observations = []
    for group in groups:
        exps = []
        for qrel_file in group:
            per_query_qrels = {}
            i = QrelsConverter(read_trec_qrels(gzip.open(qrel_file, "rt").read())).as_dict_of_dict()
            for query in list(i.keys()):
                for doc in list(i[query].keys()):
                    if i[query][doc] not in (0, 1, 2, 3):
                        i[query][doc] = 0 


            exps.append(Experiment(qrels=i))
        actual = FleissKappa().calc_agg(exps)
        assert len(actual) == 1
        observations += [actual[0].value]
    return str(int((mean(observations) * 10000))/100)

def do_agreement_calc_micro(ds, qrel_type):
    groups = grouped_qrels(ds, qrel_type)
    from topic_gen.evaluate import Experiment
    from topic_gen.evaluate.measures_agreement_multiple import FleissKappa
    from ir_measures import read_trec_qrels
    from ir_measures.util import QrelsConverter
    import gzip

    observations = []
    for group in groups:
        exps = {}
        for qrel_file in group:
            per_query_qrels = {}

            i = QrelsConverter(read_trec_qrels(gzip.open(qrel_file, "rt").read())).as_dict_of_dict()
            for query in list(i.keys()):
                for doc in list(i[query].keys()):
                    if i[query][doc] not in (0, 1, 2, 3):
                        i[query][doc] = 0 

                if query not in exps:
                    exps[query] = {}
                exps[query][str(qrel_file)] = {"oo": i[query]}

        for query in exps.keys():
            actual = FleissKappa().calc_agg([Experiment(qrels=i) for i in exps[query].values()])
            assert len(actual) == 1
            observations += [actual[0].value]
    return str(int((mean(observations) * 10000))/100)

#for ds in ["dl19", "dl20", "robust"]:
for ds in ["dl19", "dl20"]:
    ret[ds] = {}
    ret_agr[ds] = {}
    for qrel_type in ["reference", "topics-generated-title-description", "topics-generated-full", "topics-generated-title-narrative"]:
        rels = []
        print(len(list(all_qrels(ds, qrel_type))))
        for rel in all_qrels(ds, qrel_type):
            rels.append(rel)
        ret[ds][qrel_type] = str(int((mean(rels) * 10000))/100)

        #ret_agr[ds][qrel_type] = do_agreement_calc(ds, qrel_type)
        ret_agr[ds][qrel_type] = do_agreement_calc_micro(ds, qrel_type)

for k, v in [("reference", "\\cmark & \\xmark & \\xmark"), ("topics-generated-title-description", "\\cmark & \\cmark & \\xmark"), ("topics-generated-title-narrative", "\\cmark & \\xmark & \\cmark"), ("topics-generated-full", "\\cmark & \\cmark & \\cmark")]:
    l = [v, ret_agr["dl19"][k], ret_agr["dl20"][k], ret["dl19"][k], ret["dl20"][k]]
    print(" & ".join(l) + "\\\\")

