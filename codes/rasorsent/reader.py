""" Reader """

import io
import random
import uuid
import json
import torch
from codes.utils import word_tokenize, encode_answer_index, AugmentedList, Dictionary, indexes_from_sentence


class Dataset:
    def __init__(self, video_fpath, data_fpath, vocab_fpath, max_p_len, max_sent_len, shuffle=True):
        self.dictionary = Dictionary(vocab_fpath)
        self.max_p_len = max_p_len
        self.max_sent_len = max_sent_len

        # load video information
        transcripts = {}
        with open(video_fpath) as fp:
            for _video in json.load(fp):
                video_id = _video['video_id']
                transcripts[video_id] = _video['transcript']

        # load labeled data
        examples = []
        with io.open(data_fpath, encoding='utf-8') as fp:
            for datum in json.load(fp):
                transcript = transcripts[datum['video_id']]
                question = datum['question']
                answer_start = datum['answer_start']
                answer_end = datum['answer_end']
                examples.append((transcript, question, answer_start, answer_end))

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
        pids, passages, passage_masks, passage_lens, sent_lens = [], [], [], [], []
        questions, question_masks, question_lens = [], [], [],
        answers = []

        examples = self.data.next_items(batch_size)
        for transcript, question_str, answer_start, answer_end in examples:
            # indexing passages
            passage = [indexes_from_sentence(sent, self.dictionary, self.max_sent_len)
                       for sent in transcript]
            passage_len = len(transcript)
            passage_mask = [self._masking(sent) for sent in passage]
            sent_len = [min(len(word_tokenize(sent)), 35) for sent in transcript]

            # padding passage
            for _ in range(self.max_p_len - len(passage)):
                dummy_sent = indexes_from_sentence("temp", self.dictionary, self.max_sent_len)
                # dummy_sent = [self.dictionary.PAD_token] * self.max_sent_len
                passage.append(dummy_sent)
                zero_mask = [0] * self.max_sent_len
                passage_mask.append(zero_mask)
                # sent_len.append(0)  # zero-length causes a problem. for now, we give 1.
                sent_len.append(1)

            # indexing questions
            question = indexes_from_sentence(question_str, self.dictionary, self.max_sent_len)
            question_mask = self._masking(question)
            question_len = min(len(word_tokenize(question_str)), 35)

            # indexing answer
            answer = encode_answer_index(answer_start, answer_end, self.max_p_len)

            # add to batch
            pids.append(uuid.uuid4())  # Not really used
            passages.append(passage)
            passage_masks.append(passage_mask)
            passage_lens.append(passage_len)
            sent_lens.append(sent_len)
            questions.append(question)
            question_masks.append(question_mask)
            question_lens.append(question_len)
            answers.append(answer)

        # build torch.tensor
        passages = torch.tensor(passages)
        passage_masks = torch.tensor(passage_masks)
        passage_lens = torch.tensor(passage_lens)
        sent_lens = torch.tensor(sent_lens)
        questions = torch.tensor(questions)
        question_masks = torch.tensor(question_masks)
        question_lens = torch.tensor(question_lens)
        answers = torch.tensor(answers)

        if torch.cuda.is_available():
            passages = passages.long().cuda(0)
            passage_masks = passage_masks.long().cuda(0)
            passage_lens = passage_lens.long().cuda(0)
            sent_lens = sent_lens.long().cuda(0)
            questions = questions.long().cuda(0)
            question_masks = question_masks.long().cuda(0)
            question_lens = question_lens.long().cuda(0)
            answers = answers.long().cuda(0)

        return pids, passages, passage_masks, passage_lens, sent_lens, \
            questions, question_masks, question_lens, \
            answers

    def is_next_batch_available(self):
        """ """
        return self.data.is_next_batch_available()

    def reset(self, shuffle=True):
        """ """
        self.data.reset(shuffle)
