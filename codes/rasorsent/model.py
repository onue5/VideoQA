import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class OurVideoQAModel(nn.Module):
    """ Main Model """

    def __init__(self, config, emb_data):
        super(OurVideoQAModel, self).__init__()

        # Embedding layer to lookup pre-trained word embeddings
        self.embed = nn.Embedding(config.vocab_size, config.emb_dim)
        self.embed.weight.requires_grad = False  # do not propagate into the pre-trained word embeddings
        self.embed.weight.data.copy_(emb_data)

        # bi-LSTM for passage sentences

        # batch_first???
        self.psent_bilstm = nn.LSTM(
            input_size=config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            #dropout=0.75,
            batch_first=True,
            bidirectional=True
        )

        # bi-LSTM for question sentences
        self.qsent_bilstm = nn.LSTM(
            input_size=config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            # dropout=0.75,
            batch_first=True,
            bidirectional=True
        )

        # span mask
        self.span_mask = torch.ones(config.max_p_len, config.max_p_len, dtype=torch.float64)         # (max_p_len, max_p_len)
        if torch.cuda.is_available():
            self.span_mask = self.span_mask.float().cuda(0)
        self.span_mask = self.span_mask.triu()

        # span scoring
        self.fc1 = nn.Linear(6 * config.hidden_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

        parameters = filter(lambda p: p.requires_grad, self.parameters())
        for p in parameters:
            self.init_param(p)

    def _encode_sents(self, p_raw_emb, s_lens):
        """

        :param p_raw_emb: (batch_size, p_len, s_len, emb_size)
        :param s_lens: (batch_size, p_len)
        :return: p_emb: (batch_size, p_len, hidden_size * 2)
        """

        batch_size, p_len, s_len, emb_size = p_raw_emb.size()

        p_lstm_input = p_raw_emb.view(batch_size * p_len, s_len, emb_size)      # (batch_size * p_len, s_len, emb_size)
        s_lens = s_lens.view(batch_size * p_len)                                # (batch_size * p_len)

        # sort by sentence length (required by pack_padded_sequence)
        s_lens, perm_idx = s_lens.sort(0, descending=True)
        p_lstm_input = p_lstm_input[perm_idx]

        p_lstm_input_pack = pack_padded_sequence(p_lstm_input, s_lens, batch_first=True)

        lstm_out, _ = self.psent_bilstm(p_lstm_input_pack)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
                                                            # (p_len * batch_size, s_len, hidden_size * 2)

        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        s_lens = s_lens[unperm_idx]

        idx = (s_lens - 1).view(-1, 1).expand(batch_size * p_len, lstm_out.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)

        p_emb = lstm_out.gather(time_dimension, Variable(idx))
        p_emb = p_emb.squeeze(time_dimension)                       # (batch_size * p_len, hidden_size * 2)

        p_emb = p_emb.view(batch_size, p_len, -1)           # (batch_size, p_len, hidden_size * 2)

        return p_emb

    def forward(self, config, p, p_mask, p_lens, s_lens, q, q_mask, q_lens):
        """ Do forward computation

        All these inputs are of type autograd Variable

        :param config:
        :param p: (batch_size, p_len, s_len)
        :param p_mask: (batch_size, p_len, s_len)
        :param p_lens: (batch_size)
        :param s_lens: (batch_size, p_len)
        :param q:      (batch_size, max_q_len)
        :param q_mask: (batch_size, max_q_len)
        :param q_lens: (batch_size, max_q_len)
        :return:
        """

        # batch_size = config.batch_size
        batch_size = p.size(0)
        max_p_len = config.max_p_len

        # encode question
        q_emb = self.embed(q)                           # (batch_size, max_q_len, emb_dim)

        q_lens, perm_idx = q_lens.sort(0, descending=True)
        q_emb = q_emb[perm_idx]

        q_emb_pack = pack_padded_sequence(q_emb, q_lens, batch_first=True)
        lstm_out, _ = self.qsent_bilstm(q_emb_pack)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)   # (batch_size, max_q_len, hidden_size * 2)

        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        q_lens = q_lens[unperm_idx]

        idx = (q_lens - 1).view(-1, 1).expand(batch_size, lstm_out.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)

        q_emb = lstm_out.gather(time_dimension, Variable(idx))
        q_emb = q_emb.squeeze(time_dimension)                       # (batch_size, hidden_size * 2)

        # encode passage sentence
        p_raw_emb = self.embed(p)                       # (batch_size, max_p_len, s_len, emb_dim)
        p_emb = self._encode_sents(p_raw_emb, s_lens)   # (batch_size, max_p_len, 2 * hidden_size)

        # build span matrix
        _, p_len, _ = p_emb.size()

        span1 = p_emb.unsqueeze(2)                      # (batch_size, max_p_len, 1, 2 * hidden_size)
        span1 = span1.repeat(1, 1, p_len, 1)            # (batch_size, max_p_len, max_p_len, 2 * hidden_size)

        span2 = span1.transpose(1, 2)                   # (batch_size, max_p_len, max_p_len, 2 * hidden_size)

        span = torch.cat([span1, span2], dim=3)           # (batch_size, max_p_len, max_p_len, 4 * hidden_size)

        # scoring
        span = span.view(batch_size, max_p_len * max_p_len, 4 * config.hidden_dim)
                                                                 # (batch_size, max_p_len * max_p_len, 4 * hidden_size)

        q_emb = q_emb.unsqueeze(1)                                 # (batch_size, 1, 4 * hidden_size)
        q_emb = q_emb.repeat(1, max_p_len * max_p_len, 1)        # (batch_size, max_p_len * max_p_len, 2 * hidden_size)

        input = torch.cat([span, q_emb], dim=2)                  # (batch_size, max_p_len, max_p_len, 6 * hidden_size)

        scores = self.fc1(input)                                 # (batch_size, max_p_len * max_p_len)

        # Leaky Relu better than Relu and Tanh
        # Tanh is unstable -- the loss fluctuates
        # Relu fails to make the model learn for some cases -- possibly due to outputting zero for the negative region.
        scores = self.leaky_relu(scores)                         # (batch_size, max_p_len * max_p_len)
        scores = scores.view(batch_size, max_p_len, max_p_len)   # (batch_size, max_p_len, max_p_len)

        span_mask = self.span_mask.unsqueeze(0)
        span_mask = span_mask.repeat(batch_size, 1, 1)           # (batch_size, max_p_len, max_p_len)

        scores = scores * span_mask
        scores = scores.view(batch_size, max_p_len * max_p_len)  # (batch_size, max_p_len, max_p_len)

        return self.logsoftmax(scores)
        # return self.softmax(scores)

    def init_hidden(self, num_layers, hidden_dim, batch_size):
        """
        h_0: tensor containing the initial hidden state for each element in the batch
          size: (num_layers * num_directions, batch, hidden_size):
        """

        zero_t = torch.zeros(num_layers * 2, batch_size, hidden_dim)
        if torch.cuda.is_available():
            zero_t = zero_t.cuda(0)
        return (Variable(zero_t), Variable(zero_t))

    def init_param(self, param):
        if len(param.size()) < 2:
            init.uniform_(param)
        else:
            init.xavier_uniform_(param)


