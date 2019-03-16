import re
import io
import random
import unicodedata
import numpy as np
import torch


class AugmentedList:
    def __init__(self, items, shuffle_between_epoch=False):
        self.items = items
        self.cur_idx = 0
        self.shuffle_between_epoch = shuffle_between_epoch

    def is_next_batch_available(self):
        """ """
        return self.cur_idx < self.size

    def next_items(self, batch_size):
        items = self.items
        start_idx = self.cur_idx
        end_idx = start_idx + batch_size
        if end_idx <= self.size:
            # self.cur_idx = end_idx % self.size
            self.cur_idx = end_idx
            return items[start_idx: end_idx]
        else:
            self.cur_idx = self.size
            return items[start_idx:]

    def reset(self, shuffle=True):
        """ shuffle and move the index pointer to the front """

        if shuffle:
            random.shuffle(self.items)
        self.cur_idx = 0

    @property
    def size(self):
        return len(self.items)


# Helper classes
class Dictionary:
    PAD_token = 0
    UNK_token = 1

    def __init__(self, fn):
        # Intialization
        self.word2index = {}
        self.index2word = {0: "[PAD]", 1: "[UNK]"}
        self.n_words = 2

        # Load words from file
        f = io.open(fn, encoding='utf-8')
        for word in f:
            self.add_word(word.strip())
        f.close()

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def indexes_from_sentence(sentence, dictionary, max_len):
    word2index = dictionary.word2index
    PAD_token = dictionary.PAD_token
    UNK_token = dictionary.UNK_token
    indexes = []
    words = word_tokenize(sentence)
    for word in words[:max_len]:
        if len(word) > 0:
            if word in word2index:
                indexes.append(word2index[word])
            else:
                indexes.append(UNK_token)

    while len(indexes) < max_len:
        indexes.append(PAD_token)
    return indexes


def one_hot_encoding(num_classes, idx):
    encoding = [0] * num_classes
    encoding[idx] = 1
    return encoding


def load_glove_vectors(path, dictionary, emb_size=300):
    vocab_size = dictionary.n_words
    word2index = dictionary.word2index
    weights = np.random.randn(vocab_size, emb_size)

    with io.open(path, encoding='utf-8') as f:
        for line in f:
            word, emb = line.strip().split(' ', 1)
            if word in word2index:
                weights[word2index[word]] = np.asarray(list(map(float, emb.split(' ')))[:emb_size])

    emb = torch.from_numpy(weights)
    return emb


def unicode_to_ascii(s):
    # s = unicode(s)
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^0-9a-zA-Z]+", r" ", s)
    s = s.strip()
    return s


def word_tokenize(sent):
    sent = normalize_string(sent)
    return sent.split(' ')


def measure_diff(a, a_hats, max_p_len):
    """

    :param a: (batch size)
    :param a_hats: (batch size)
    :param max_p_len:
    :return: diff: (batch size)
    """

    gt_start_index = a / max_p_len
    gt_end_index = a % max_p_len

    pred_start_index = a_hats / max_p_len
    pred_end_index = a_hats % max_p_len

    # temporary
    # diff = abs(gt_start_index - pred_start_index) + abs(gt_end_index - pred_end_index)

    diff = abs(gt_start_index - pred_start_index)

    return diff


def encode_answer_index(ans_start_word_idx, ans_end_word_idx, max_p_len):

    # assert ans_end_word_idx - ans_start_word_idx + 1 <= max_ans_len
    return ans_start_word_idx * max_p_len + (ans_end_word_idx - ans_start_word_idx)

