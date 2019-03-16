""" Reader """

import io
import random
import uuid
import json
from collections import defaultdict
import torch
from codes.utils import word_tokenize, AugmentedList, Dictionary, indexes_from_sentence


class Dataset(object):
    def __init__(self, video_fpath, data_fpath, vocab_fpath, max_q_len, max_p_len, shuffle=True):

        self.dictionary = Dictionary(vocab_fpath)
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len

        # load video information
        videos = defaultdict(list)
        with open(video_fpath) as fp:
            for video in json.load(fp):
                transcript = video["transcript"]
                segments = video["segments"]
                for segment in segments:
                    video_id = video["video_id"]
                    start_index = segment["sentence_indexes"]["start"]
                    end_index = segment["sentence_indexes"]["end"]
                    segment_txt = " ".join(transcript[start_index:end_index])
                    videos[video_id].append({
                        "start_index": start_index,
                        "end_index": end_index,
                        "text": segment_txt
                    })

        # load labeled data
        examples = []
        with io.open(data_fpath, encoding='utf-8') as fp:
            for datum in json.load(fp):
                video_id = datum['video_id']
                question = datum['question']
                answer_start = datum['answer_start']
                answer_end = datum['answer_end']

                for i, segment in enumerate(videos[video_id]):
                    score = 1 if answer_start == segment["start_index"] and answer_end == segment["end_index"] \
                        else 0

                    # if video_id == "19194" and question == "what does adjusting colors in an image do?":
                    #     print(question, score)

                    examples.append((video_id, question, segment["text"], score))

        if shuffle:
            random.shuffle(examples)
        self.data = AugmentedList(examples)

    def _shorten_sent(self, sent):
        """ """
        tokens = word_tokenize(sent)
        new_sent = " ".join(tokens[:self.max_sent_len])
        return new_sent

    def _masking(self, sent):
        mask = [1 if self.dictionary.PAD_token != index else 0
                for index in sent]
        return mask

    def batch_num(self, batch_size):
        """ """
        return int(self.data.size / batch_size)

    def next_batch(self, batch_size):
        """ """
        video_ids, question_strs = [], [],
        questions, question_masks, question_lens = [], [], [],
        passages, passage_masks, passage_lens = [], [], [],
        scores = []

        examples = self.data.next_items(batch_size)
        for video_id, question_str, passage_str, score in examples:

            video_ids.append(video_id)
            question_strs.append(question_str)

            # indexing passages
            passage = indexes_from_sentence(passage_str, self.dictionary, self.max_p_len)
            passage_mask = self._masking(passage)
            passage_len = min(len(word_tokenize(passage_str)), self.max_p_len)

            # indexing questions
            question = indexes_from_sentence(question_str, self.dictionary, self.max_q_len)
            question_mask = self._masking(question)
            question_len = min(len(word_tokenize(question_str)), self.max_q_len)

            # add to batch
            passages.append(passage)
            passage_masks.append(passage_mask)
            passage_lens.append(passage_len)
            questions.append(question)
            question_masks.append(question_mask)
            question_lens.append(question_len)
            scores.append(score)

        # build torch.tensor
        passages = torch.tensor(passages)
        passage_masks = torch.tensor(passage_masks)
        passage_lens = torch.tensor(passage_lens)
        questions = torch.tensor(questions)
        question_masks = torch.tensor(question_masks)
        question_lens = torch.tensor(question_lens)
        scores = torch.tensor(scores)

        if torch.cuda.is_available():
            passages = passages.long().cuda(0)
            passage_masks = passage_masks.long().cuda(0)
            passage_lens = passage_lens.long().cuda(0)
            questions = questions.long().cuda(0)
            question_masks = question_masks.long().cuda(0)
            question_lens = question_lens.long().cuda(0)
            scores = scores.long().cuda(0)

        return video_ids, question_strs, \
               questions, question_masks, question_lens, \
               passages, passage_masks, passage_lens, \
               scores

    def is_next_batch_available(self):
        """ """
        return self.data.is_next_batch_available()

    def reset(self, shuffle=True):
        """ """
        self.data.reset(shuffle)
