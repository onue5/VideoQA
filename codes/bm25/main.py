""" BM25-based


measure map, mrr, top1 acc
"""

import os
import json
import elasticsearch
import numpy as np

# data dir
DATA_DIR = "../../data/"

# elasticsearch
es = elasticsearch.Elasticsearch(hosts=[{'host': "localhost"}])


def _retrieve(test):
    """ """

    query = test["question"]
    video_id = test['video_id']

    body = {
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "text": query
                    }
                },
                "filter": {
                    "term": {
                        "video_id": video_id
                    }
                }
            }
        }
    }

    res = es.search(index="videoqa", body=body, size=20)
    preds = [(hit['_source']['start_index'], hit['_source']['end_index'])
             for hit in res['hits']['hits']]

    return preds


def _measure_metrics(tests, preds):
    """ """

    # map is equal to mrr when only one relevant document exists
    # Formula = mean(1 / rank of relevant document)

    # mrr
    scores = []
    for test, pred in zip(tests, preds):
        gt_start, gt_end = test['answer_start'], test['answer_end']
        try:
            rank = pred.index((gt_start, gt_end))
            scores.append(1/float(rank + 1))
        except ValueError:
            scores.append(0)
    mrr = np.mean(scores)

    # acc
    n_cor = 0
    for test, pred in zip(tests, preds):
        gt_start, gt_end = test['answer_start'], test['answer_end']
        if len(pred) >= 1:
            pred_start, pred_end = pred[0]
            if pred_start == gt_start and pred_end == gt_end:
                n_cor += 1
    top1_acc = float(n_cor) / len(tests)

    print("mrr: {:4f}    acc: {:4f}".format(mrr, top1_acc))


def _run():

    with open(os.path.join(DATA_DIR, "test.json")) as fp:
        tests = json.load(fp)

    preds = [_retrieve(test) for test in tests]
    _measure_metrics(tests, preds)


if __name__ == "__main__":
    _run()
