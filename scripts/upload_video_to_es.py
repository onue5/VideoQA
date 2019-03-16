""" Upload video segment transcript to ES


"""

import json
import elasticsearch


es = elasticsearch.Elasticsearch(hosts=[{'host': "localhost"}])

ES_INDEX_NAME = "videoqa"

# reset ES
if es.indices.exists(index=ES_INDEX_NAME):
    es.indices.delete(index=ES_INDEX_NAME)

with open('../data/videos.json') as fp:
    videos = json.load(fp)

    for video in videos:

        transcript = video["transcript"]
        segments = video["segments"]
        for segment in segments:
            start_index = segment["sentence_indexes"]["start"]
            end_index = segment["sentence_indexes"]["end"]
            segment_txt = " ".join(transcript[start_index:end_index])
            es_doc = {
                "video_id": video["video_id"],
                "start_index": start_index,
                "end_index": end_index,
                "text": segment_txt
            }

            es.index(index=ES_INDEX_NAME, doc_type='video_segment', body=es_doc)
