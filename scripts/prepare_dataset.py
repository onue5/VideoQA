""" Prepare two datasets -- videos.json and qa.json.
 * videos.json: information about each video
 * qa.json: information about qa pairs

Usage:
    python prepare_dataset.py
"""
import sys
sys.path.append('../../VideoQAModelDS/')
import csv
import io
import os
import json
import random
from collections import defaultdict
from codes.utils import word_tokenize

# data directory
DATA_DIR = "../data/"

# TRAIN/DEV/TEST RATIO
TRAIN_RATIO = .6
DEV_RATIO = .2
TEST_RATIO = .2

# glove path
GLOVE_FPATH = '/mnt/ilcompf0d1/user/dkim/Data/Embeddings/glove.840B.300d.txt'


def _load_answer_segments():
    """ Load answer segmentation information (created by Anthony Colas) from  full_file_annotated.csv
    """

    videoid2segments = defaultdict(list)

    with open(os.path.join(DATA_DIR, "raw/full_file_annotated.csv")) as fp:
        csvread = csv.reader(fp)
        next(csvread)
        for index, answer, t0, t1, webm_url, video_id in csvread:
            videoid2segments[video_id].append((answer, t0, t1))

    return videoid2segments


def _build_timestamp2sentindex(full_video):
    """ From Seokhwan's video json file, build the mappings from the timestamps to the sentence indexes
    """

    timestamp2sentindex = {}
    sentindex = 0
    for segment in full_video['segments']:
        for sent_info in segment['sentences']:
            if sent_info['begin']:
                start_timestamp = float(sent_info['begin'])
                timestamp2sentindex[start_timestamp] = sentindex
            if sent_info['end']:
                end_timestamp = float(sent_info['end'])
                timestamp2sentindex[end_timestamp] = sentindex

            sentindex += 1

    return timestamp2sentindex


def _write_video_file():
    """ Generate the video file (./data/videos.json)

    Each line in the video file has the following json:
        {
            video_id: <video id>,
            video_title: <video title>,
            video_url: <video url>
            transcript: ["sent1", "sent2", ...],
            segments: [
                {
                    title: "title of the segment"
                    timestamps:
                        start: <start timestamp>,
                        end: <end timestamp>,
                    sentence_indexes:
                        start: <start index>,
                        end_index: <end index>
                }
            ]
        ]
    """

    # load the full video information
    with open(os.path.join(DATA_DIR, "raw/helpx_photoshop_tutorials.events.json")) as fp:
        full_videos = json.load(fp)

    # load the segmentation information
    videoid2segments = _load_answer_segments()

    # generate video information
    videos = []
    for full_video in full_videos:
        video_id = full_video['video_id']
        transcript = [sent_info['sent']
                      for segment in full_video['segments']
                      for sent_info in segment['sentences']]

        timestamp2sentindex = _build_timestamp2sentindex(full_video)
        segments = []

        for segment_title, start_timestamp, end_timestamp in videoid2segments[video_id]:
            segments.append({
                "title": segment_title,
                "timestamps": {
                    "start": start_timestamp,
                    "end": end_timestamp
                },
                "sentence_indexes": {
                    "start": timestamp2sentindex[float(start_timestamp)],
                    "end": timestamp2sentindex[float(end_timestamp)]
                }
            })

        video = {
            "video_id": video_id,
            "video_title": full_video['video_title'],
            "video_url": full_video['video_url'],
            "transcript": transcript,
            "segments": segments
        }
        videos.append(video)

    with open(os.path.join(DATA_DIR, "videos.json"), 'w') as fp:
        json.dump(videos, fp, indent=4)


def _load_answer_url2segment_info():
    """ """
    answer_url2segment_info = {}
    with open(os.path.join(DATA_DIR, "videos.json")) as fp:
        videos = json.load(fp)
        for video in videos:
            for segment_json in video["segments"]:
                answer_url = "{}#t={},{}".format(
                    video["video_url"],
                    segment_json['timestamps']['start'],
                    segment_json['timestamps']['end']
                )
                segment_info = {
                    'video_id': video['video_id'],
                    'answer_start': segment_json['sentence_indexes']['start'],
                    'answer_end': segment_json['sentence_indexes']['end']
                }
                answer_url2segment_info[answer_url] = segment_info
    #
    # for answer_url in sorted(answer_url2segment_info.keys()):
    #     print(answer_url)

    return answer_url2segment_info


def _write_qa_file():
    """ Generate the qa annotation file (./data/qa.json)

    Each line in the annotation file has the following json:
        {
            video_id: <video id>,
            question: "question",
            answer_start: <start sentence index>
            answer_end: <end sentence index>
        }
    """

    answer_url2segment_info = _load_answer_url2segment_info()

    data = []
    with open(os.path.join(DATA_DIR, "raw/question_results_with_paraphrases.json")) as fp:
        qa_json = json.load(fp)

        for answer_url, question_infos in qa_json.items():
            segment_info = answer_url2segment_info[answer_url]

            for question_info in question_infos:
                for question in question_info['inputQuestion']:
                    data.append({
                        "video_id": segment_info['video_id'],
                        "question": question.lower(),
                        "answer_start": segment_info['answer_start'],
                        "answer_end": segment_info['answer_end']
                    })

    print("total data size: ", len(data))
    data.sort(key=lambda x: x['video_id'])

    with open(os.path.join(DATA_DIR, "qa.json"), 'w') as fp:
        json.dump(data, fp, indent=4)

    return data


def _split_data():
    """ """

    with open(os.path.join(DATA_DIR, "qa.json")) as fp:
        qa_data = json.load(fp)

    random.shuffle(qa_data)

    train_end_index = int(len(qa_data) * TRAIN_RATIO)
    dev_end_index = int(len(qa_data) * (TRAIN_RATIO + DEV_RATIO))

    train = qa_data[:train_end_index]
    dev = qa_data[train_end_index+1:dev_end_index]
    test = qa_data[dev_end_index+1:]

    for _data, fname in [(train, "train.json"), (dev, "dev.json"), (test, "test.json")]:
        with open(os.path.join(DATA_DIR, fname), 'w') as fp:
            json.dump(_data, fp, indent=4)
            print("the size of {}: {}".format(fname[:-4], len(_data)))


def _add_to_vocab(sent, all_words):
    """ add words to vocabulary"""
    tokens = word_tokenize(sent)
    for token in tokens:
        all_words.add(token)


def _generate_vocab():
    """ """

    all_words = set([])

    # load vocabularies from videos.json
    with open(os.path.join(DATA_DIR, "videos.json")) as fp:
        video_data = json.load(fp)
        for one_video in video_data:
            for sent in one_video["transcript"]:
                _add_to_vocab(sent, all_words)

    # load vocabularies from train/dev/test.json
    for fname in ["train.json", "dev.json", "test.json"]:
        with open(os.path.join(DATA_DIR, fname)) as fp:
            _data = json.load(fp)

        for _inst in _data:
            _add_to_vocab(
                sent=_inst["question"].strip(),
                all_words=all_words
            )

    with io.open(GLOVE_FPATH, encoding='utf-8') as in_fp, \
            io.open(os.path.join(DATA_DIR, 'vocab.txt'), 'w') as vocab_fp, \
            io.open(os.path.join(DATA_DIR, 'vocab_embedding.txt'), 'w') as embed_fp:

        for line in in_fp:
            word, emb = line.strip().split(' ', 1)
            if word in all_words:
                vocab_fp.write('%s\n'%word)
                embed_fp.write(line)


def run():
    print("1. create videos.json and qa.json")
    _write_video_file()
    _write_qa_file()

    print("2. create train.json, dev.json, test.json")
    _split_data()

    print("3. create vocab.txt and vocab_embedding.txt")
    _generate_vocab()


if __name__ == "__main__":
    run()
