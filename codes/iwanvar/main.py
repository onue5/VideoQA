""" """
import time
import torch
from collections import defaultdict
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from codes.service.model import VideoQAServiceModel
from codes.service.reader import Dataset
from codes.utils import load_glove_vectors


class Config(object):
    def __init__(self):
        self.name = "VideoQAService"

        self.max_q_len = 35             # maximum question length
        self.max_t_len = 35             # maximum title length
        self.max_p_len = 1000           # maximum segment length

        self.kernel_sizes = [3, 4, 5]   # CNN kernel size
        self.kernel_num = 10            # CNN kernel num

        self.emb_dim = 300              # dimension of word embeddings
        self.lstm_hidden_dim = 100      # dimension of hidden state of each uni-directional LSTM
        self.att_hidden_dim = 50        # dimension of intermediate rep. for computing attention weights
        self.score_dim = 50             # dimension of intermediate rep. for computing scoring
        self.num_layers = 2             # number of BiLSTM layers, where BiLSTM is applied
        self.batch_size = 30

        self.seed = np.random.random_integers(1e6, 1e9)

    def __repr__(self):
        ks = sorted(k for k in self.__dict__ if k not in ['name'])
        return '\n'.join('{:<30s}{:<s}'.format(k, str(self.__dict__[k])) for k in ks)


# config
config = Config()

# a loss function (negative log-likelihood)
# loss_function = nn.NLLLoss()
loss_function = nn.CrossEntropyLoss()

# loss_function = nn.MSELoss()
if torch.cuda.is_available():
    loss_function = loss_function.cuda(0)


def _eval_dev(model, dev_dataset):
    """ """
    losses = []
    dev_acces = []

    dev_dataset.reset(shuffle=False)

    # collect the scores for each pair of (question, segment)
    score_results = defaultdict(list)
    while dev_dataset.is_next_batch_available():
        batch = dev_dataset.next_batch(config.batch_size)
        video_ids, q_strs, qs, q_masks, q_lens, ts, t_masks, t_lens, ps, p_masks, p_lens, batch_a = batch

        model.eval()
        pred_scores = model(
            config,
            Variable(qs, requires_grad=False), Variable(q_masks, requires_grad=False),
            Variable(q_lens, requires_grad=False),
            Variable(ts, requires_grad=False), Variable(t_masks, requires_grad=False),
            Variable(t_lens, requires_grad=False),
            Variable(ps, requires_grad=False), Variable(p_masks, requires_grad=False),
            Variable(p_lens, requires_grad=False),
        )
        # print(pred_scores)
        loss = loss_function(pred_scores, batch_a)
        _, a_hats = torch.max(pred_scores, 1)
        losses.append(loss.item())

        # acc = torch.eq(a_hats, batch_a).float().mean()
        acc = torch.eq(a_hats, batch_a).float().sum()
        dev_acces.append(acc)

        for video_id, q_str, pred_score, a in zip(video_ids, q_strs, pred_scores, batch_a):
            # print(q_str, pred_score, a)
            score_results[(video_id, q_str)].append({
                "pred_score": pred_score[1].item(),
                "gt_answer": a
            })

    dev_mrrs = []

    n_tot = 0
    n_miss = 0
    for (video_id, qstr), score_result in score_results.items():
        sorted_score_result = sorted(score_result, key=lambda x: x["pred_score"], reverse=True)
        gt_answers = [x['gt_answer'] for x in sorted_score_result]

        n_tot += 1
        try:
            gt_rank = gt_answers.index(1) + 1
            mrr = 1 / float(gt_rank)
            dev_mrrs.append(mrr)
        except ValueError:
            # print(video_id, qstr, score_result)
            n_miss += 1

    # print("n_tot: {}".format(n_tot))
    # print("n_miss: {}".format(n_miss))

    dev_loss = np.average(losses)
    dev_acc = sum(dev_acces) / dev_dataset.size
    dev_mrr = np.average(dev_mrrs)

    return dev_loss, dev_acc, dev_mrr


def _train(model, train_dataset, epochid):
    """ """

    losses = []
    acces = []

    # lr = 0.001
    lr = 0.1
    if epochid % 10 == 0:
        lr = lr * 0.95

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr)

    train_dataset.reset()

    while train_dataset.is_next_batch_available():
        batch = train_dataset.next_batch(config.batch_size)
        video_ids, q_strs, qs, q_masks, q_lens, ts, t_masks, t_lens, ps, p_masks, p_lens, a = batch

        model.zero_grad()

        model.train()
        scores = model(
            config,
            Variable(qs, requires_grad=False), Variable(q_masks, requires_grad=False),
            Variable(q_lens, requires_grad=False),
            Variable(ts, requires_grad=False), Variable(t_masks, requires_grad=False),
            Variable(t_lens, requires_grad=False),
            Variable(ps, requires_grad=False), Variable(p_masks, requires_grad=False),
            Variable(p_lens, requires_grad=False),
        )
        loss = loss_function(scores, a)
        _, a_hats = torch.max(scores, 1)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        losses.append(loss.item())
        # acc = torch.eq(a_hats, a).float().mean()
        acc = torch.eq(a_hats, a).float().sum()
        acces.append(acc)

    trn_loss = np.sum(losses)
    # trn_acc = np.average(acces)
    trn_acc = sum(acces) / train_dataset.size

    return trn_loss, trn_acc


def train(video_fpath, train_fpath, dev_fpath, vocab_fpath, embedding_fpath, num_epoch):
    """     """
    trn_dataset = Dataset(
        video_fpath, train_fpath, vocab_fpath, config.max_q_len, config.max_t_len, config.max_p_len)
    dev_dataset = Dataset(
        video_fpath, dev_fpath, vocab_fpath, config.max_q_len, config.max_t_len, config.max_p_len, shuffle=True)

    dictionary = trn_dataset.dictionary
    embedding_dim = config.emb_dim

    emb = load_glove_vectors(embedding_fpath, dictionary, emb_size=embedding_dim)
    config.vocab_size = dictionary.n_words

    model = VideoQAServiceModel(config, emb)
    if torch.cuda.is_available():
        model = model.cuda()

    time1 = time.time()
    print("# of batches: {}".format(trn_dataset.batch_num(config.batch_size)))
    for epoch in range(1, num_epoch+1):
        trn_loss, trn_acc = _train(model, trn_dataset, epoch)
        dev_loss, dev_acc, dev_mrr = _eval_dev(model, dev_dataset)

        print("Epoch {}, TRN LOSS: {:.4f}    TRN ACC: {:.4f}".format(epoch, trn_loss, trn_acc))
        print("Epoch {}, DEV LOSS: {:.4f}    DEV ACC: {:.4f}        DEV_MRR: {:.4f}".format(
            epoch, dev_loss, dev_acc, dev_mrr))
        print("time taken: {:.4}".format(time.time() - time1))
        print()
