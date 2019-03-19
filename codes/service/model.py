import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VideoQAServiceModel(nn.Module):
    """ Main Squad Model """

    def __init__(self, config, emb_data):
        super(VideoQAServiceModel, self).__init__()

        # Embedding layer to lookup pre-trained word embeddings
        self.embed = nn.Embedding(config.vocab_size, config.emb_dim)
        self.embed.weight.requires_grad = False  # do not propagate into the pre-trained word embeddings
        self.embed.weight.data.copy_(emb_data)

        self._bilstm = nn.LSTM(
            input_size=config.emb_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.num_layers,
            dropout=0.5,
            batch_first=True,
            bidirectional=True
        )

        self._convs = nn.ModuleList(
            [nn.Conv2d(1, config.kernel_num, (k, config.emb_dim))
             for k in config.kernel_sizes]
        )

        self._att_affine1 = nn.Linear(2 * config.lstm_hidden_dim, config.att_hidden_dim)
        self._att_affine2 = nn.Linear(config.att_hidden_dim, 1)

        self._drop_layer = nn.Dropout(p=.5)

        self._score_affine1 = nn.Linear(
            4 * 2 * config.lstm_hidden_dim + config.kernel_num * len(config.kernel_sizes),
            # 4 * 2 * config.lstm_hidden_dim,
            config.score_dim
        )
        self._score_affine2 = nn.Linear(config.score_dim, 2)

        parameters = filter(lambda p: p.requires_grad, self.parameters())
        for p in parameters:
            self.init_param(p)

    def _encode_short_text(self, sent, sent_mask, sent_lens):
        """ """

        # Encode via bi-LSTM
        sent_emb = self.embed(sent)                                     # (batch_size, max_sent_len, emb_dim)

        sent_lens, perm_idx = sent_lens.sort(0, descending=True)
        sent_emb = sent_emb[perm_idx]

        sent_emb_pack = pack_padded_sequence(sent_emb, sent_lens, batch_first=True)
        lstm_out, _ = self._bilstm(sent_emb_pack)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)   # (batch_size, max_sent_len, hidden_size * 2)

        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]

        # Self-weight
        self_weights = self._att_affine1(lstm_out)                      # (batch_size, q_len, att_hidden_dim)
        self_weights = F.tanh(self_weights)
        self_weights = self._att_affine2(self_weights)                  # (batch_size, q_len, 1)

        # self_weights = F.multiply(self_weights, sent_mask)
        self_weights = F.softmax(self_weights, dim=1)                   # (batch_size, q_len, 1)
        sent = self_weights * lstm_out                                  # (batch_size, q_len, 2 * hidden_dim)
        sent = sent.sum(dim=1)                                          # (batch_size, 2 * hidden_dim)
        return sent

    def _encode_transcript(self, p, p_mask, p_lens):
        """

        :param p:       (batch_size, max_p_len)
        :param p_mask:
        :param p_lens:
        :return:
        """

        p = self.embed(p)                              # (batch_size, max_p_len, emb_dim)
        p = p.unsqueeze(dim=1)                         # (batch_size, 1, max_p_len, emb_dim)
        p = [F.relu(conv(p)).squeeze(3)
             for conv in self._convs]                  # [(batch_size, kernel_num, max_p_len), ...] * len(kernel_sizes)
        p = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in p]                               # [(batch_size, kernel_num), ...] * len(kernel_sizes)
        p = torch.cat(p, 1)
        return p

    def forward(self, config, q, q_mask, q_lens, t, t_mask, t_lens, p, p_mask, p_lens):
        """ Do forward computation

        All these inputs are of type autograd Variable

        :param config:
        :param q:      (batch_size, max_q_len)
        :param q_mask: (batch_size, max_q_len)
        :param q_lens: (batch_size)
        :param t:      (batch_size, max_t_len)
        :param t_mask: (batch_size, max_t_len)
        :param t_lens: (batch_size)
        :param p:      (batch_size, max_p_len)
        :param p_mask: (batch_size, max_p_len)
        :param p_lens: (batch_size)
        :return:
        """

        q = self._encode_short_text(q, q_mask, q_lens)             # (batch_size, 2 * hidden_size)
        t = self._encode_short_text(t, t_mask, t_lens)             # (batch_size, 2 * hidden_size)
        p = self._encode_transcript(p, p_mask, p_lens)             # (batch_size, 2 * hidden_size)

        h_cross = q * t
        h_plus = torch.abs(q - t)

        inp = torch.cat([q, t, h_cross, h_plus, p], dim=1)
        inp = self._drop_layer(inp)

        scores = F.tanh(self._score_affine1(inp))
        scores = F.tanh(self._score_affine2(scores))

        return scores

    def init_param(self, param):
        if len(param.size()) < 2:
            init.uniform_(param)
        else:
            init.xavier_uniform_(param)


