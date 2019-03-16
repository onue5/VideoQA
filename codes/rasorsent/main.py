""" """
import time
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from codes.rasorsent.model import OurVideoQAModel
from codes.rasorsent.reader import Dataset
from codes.utils import load_glove_vectors, measure_diff


class Config(object):
    def __init__(self):
        self.name = "SimplifiedRasorVideoQAModel"

        self.max_p_len = 186       # maximum passage length
        self.max_sent_len = 35           # maximum sentence length
        # self.max_ans_len = 40  # maximal answer length, answers of longer length are discarded
        self.emb_dim = 300  # dimension of word embeddings
        self.ff_dim = 100
        # self.batch_size = 20
        self.batch_size = 10
        self.num_layers = 2  # number of BiLSTM layers, where BiLSTM is applied

        # changed from 100 to 50
        self.hidden_dim = 100  # dimension of hidden state of each uni-directional LSTM
        # vocab size for config
        # self.vocab_size = 2606
        # self.vocab_size = 3593
        self.vocab_size = None
        self.seed = np.random.random_integers(1e6, 1e9)

    def __repr__(self):
        ks = sorted(k for k in self.__dict__ if k not in ['name'])
        return '\n'.join('{:<30s}{:<s}'.format(k, str(self.__dict__[k])) for k in ks)


# config
config = Config()

# tolerance window sizes
TOLERANCE_WINDOWS = [0, 2, 4, 6, 8, 10]

# a loss function (negative log-likelihood)
loss_function = nn.NLLLoss()
if torch.cuda.is_available():
    loss_function = loss_function.cuda(0)


def _eval_dev(model, dev_dataset, verbose=False):
    """ """

    losses = []
    dev_acces = {window: [] for window in TOLERANCE_WINDOWS}

    dev_dataset.reset(shuffle=False)
    # print("# of batches: {}".format(dev_dataset.batch_num(config.batch_size)))

    while dev_dataset.is_next_batch_available():
        batch = dev_dataset.next_batch(config.batch_size)
        _, p, p_mask, p_len, s_lens, q, q_mask, q_len, a = batch

        scores = model(
            config, Variable(p, requires_grad=False), Variable(p_mask, requires_grad=False),
            Variable(p_len, requires_grad=False), Variable(s_lens, requires_grad=False),
            Variable(q, requires_grad=False), Variable(q_mask, requires_grad=False),
            Variable(q_len, requires_grad=False)
        )

        loss = loss_function(scores, a)
        _, a_hats = torch.max(scores, 1)

        # temporary: for printing the prediction
        if verbose:
            max_p_len = config.max_p_len
            gt_start_index = a / max_p_len
            gt_end_index = a % max_p_len

            pred_start_index = a_hats / max_p_len
            pred_end_index = a_hats % max_p_len

            for i in range(a.size()[0]):
                print("pred: ({}, {}), gt: ({}, {})".format(
                    pred_start_index[i], pred_end_index[i],
                    gt_start_index[i], gt_end_index[i]))

        diff = measure_diff(a, a_hats, config.max_p_len)
        batch_tolerance_acc = {window: (diff <= window).float()
                               for window in TOLERANCE_WINDOWS}

        for window in TOLERANCE_WINDOWS:
            batch_results = batch_tolerance_acc[window].cpu().numpy().tolist()
            dev_acces[window].extend(batch_results)
        losses.append(loss.item())

    dev_loss = np.average(losses)
    dev_acc = {window: np.average(acc)
               for window, acc in dev_acces.items()}

    return dev_loss, dev_acc


def _train(model, train_dataset, epochid):
    """ """

    losses = []
    trn_acces = {window: [] for window in TOLERANCE_WINDOWS}

    # lr = 0.001
    lr = 0.01
    if epochid % 10 == 0:
        lr = lr * 0.95

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr)

    train_dataset.reset()

    while train_dataset.is_next_batch_available():
        batch = train_dataset.next_batch(config.batch_size)
        _, p, p_mask, p_len, s_lens, q, q_mask, q_len, a = batch

        model.zero_grad()

        scores = model(
            config, Variable(p, requires_grad=False), Variable(p_mask, requires_grad=False),
            Variable(p_len, requires_grad=False), Variable(s_lens, requires_grad=False),
            Variable(q, requires_grad=False), Variable(q_mask, requires_grad=False),
            Variable(q_len, requires_grad=False)
        )

        loss = loss_function(scores, a)
        _, a_hats = torch.max(scores, 1)
        # calculate the tolerance metric
        diff = measure_diff(a, a_hats, config.max_p_len)
        batch_tolerance_acc = {window: (diff <= window).float()
                               for window in TOLERANCE_WINDOWS}

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        for window in TOLERANCE_WINDOWS:
            batch_results = batch_tolerance_acc[window].cpu().numpy().tolist()
            trn_acces[window].extend(batch_results)
        losses.append(loss.item())

    # print("losses", losses)
    trn_loss = np.average(losses)

    trn_acc = {window: np.average(acc)
               for window, acc in trn_acces.items()}

    return trn_loss, trn_acc


def train(model_name, video_fpath, train_fpath, dev_fpath, vocab_fpath, output_dir, embedding_fpath, num_epoch):
    """     """
    train_dataset = Dataset(video_fpath, train_fpath, vocab_fpath,
                            config.max_p_len, config.max_sent_len)
    dev_dataset = Dataset(video_fpath, dev_fpath, vocab_fpath,
                          config.max_p_len, config.max_sent_len, shuffle=False)

    dictionary = train_dataset.dictionary
    embedding_dim = config.emb_dim

    emb = load_glove_vectors(embedding_fpath, dictionary, emb_size=embedding_dim)
    config.vocab_size = dictionary.n_words

    model = OurVideoQAModel(config, emb)
    if torch.cuda.is_available():
        model = model.cuda()

    trn_format = "Epoch {:2d} TRN loss:{:.4f}   " + ", ".join(["acc@{}: {{:.4f}}".format(window)
                                                               for window in TOLERANCE_WINDOWS])
    dev_format = "Epoch {:2d} DEV loss:{:.4f}   " + ", ".join(["acc@{}: {{:.4f}}".format(window)
                                                               for window in TOLERANCE_WINDOWS])

    time1 = time.time()
    print("# of batches: {}".format(train_dataset.batch_num(config.batch_size)))
    for epoch in range(1, num_epoch+1):
        trn_loss, trn_acc = _train(model, train_dataset, epoch)
        dev_loss, dev_acc = _eval_dev(model, dev_dataset, verbose=False)

        trn_acc_tuple = (trn_acc[window] for window in TOLERANCE_WINDOWS)
        dev_acc_tuple = (dev_acc[window] for window in TOLERANCE_WINDOWS)

        print(trn_format.format(epoch, trn_loss, *trn_acc_tuple))
        print(dev_format.format(epoch, dev_loss, *dev_acc_tuple))
        print("time taken: {:.4}".format(time.time() - time1))
        print()

        # log = "Epoch {:2d} Train loss:{:.6f} acc@0:{:.4f}, acc@1:{:.4f}, acc@3:{:.4f}, acc@5:{:.4f}  " \
        #       "Dev loss:{:.6f} acc@0:{:.4f}, acc@1:{:.4f}, acc@3:{:.4f}, acc@5:{:.4f}".\
        #     format(epoch, trn_loss, trn_acc[0], trn_acc[1], trn_acc[3], trn_acc[5],
        #            dev_loss, dev_acc[0], dev_acc[1], dev_acc[3], dev_acc[5])
        # print(log)
