""" Inspect dataset

Look at the distribution of the answer length
"""

import json
from collections import Counter


with open('../data/videos.json') as fp:
    _data = json.load(fp)
    print(len(_data))
    print(max([len(datum['transcript']) for datum in _data]))

    seg_num = sum([len(datum["segments"]) for datum in _data])
    print(seg_num)

