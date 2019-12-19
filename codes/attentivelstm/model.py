import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentiveLSTMVideoQAModel(nn.Module):
    """ Main Model """

    def __init__(self, config, emb_data):
        super(AttentiveLSTMVideoQAModel, self).__init__()

        # Embedding layer to lookup pre-trained word embeddings
        self.embed = nn.Embedding(config.vocab_size, config.emb_dim)
        self.embed.weight.requires_grad = False  # do not propagate into the pre-trained word embeddings
        self.embed.weight.data.copy_(emb_data)

        self._bilstm = nn.LSTM(
            input_size=config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=0.5,
            batch_first=True,
            bidirectional=True
        )

        # attention weights
        self.w_qm = nn.Linear(2 * config.hidden_dim, config.att_dim, bias=False)
        self.w_pm = nn.Linear(2 * config.hidden_dim, config.att_dim, bias=False)
        self.w_ms = nn.Linear(config.att_dim, 1, bias=False)

        self.att_softmax = nn.Softmax(dim=1)
        self.cosine = nn.CosineSimilarity()

        self.ff = nn.Linear(4 * config.hidden_dim, 1, bias=False)

        parameters = filter(lambda p: p.requires_grad, self.parameters())
        for p in parameters:
            self.init_param(p)

    def _bilstm_encode(self, sent, sent_lens):
        """ """

        sent_emb = self.embed(sent)                                     # (batch_size, max_sent_len, emb_dim)

        sent_lens, perm_idx = sent_lens.sort(0, descending=True)
        sent_emb = sent_emb[perm_idx]

        sent_emb_pack = pack_padded_sequence(sent_emb, sent_lens, batch_first=True)
        lstm_out, _ = self._bilstm(sent_emb_pack)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)   # (batch_size, max_sent_len, hidden_size * 2)

        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        sent_lens = sent_lens[unperm_idx]

        return lstm_out

    def forward(self, config, q, q_mask, q_lens, p, p_mask, p_lens):
        """ Do forward computation

        All these inputs are of type autograd Variable

        :param config:
        :param q:      (batch_size, max_q_len)
        :param q_mask: (batch_size, max_q_len)
        :param q_lens: (batch_size, max_q_len)
        :param p:      (batch_size, max_p_len)
        :param p_mask: (batch_size, max_p_len)
        :param p_lens: (batch_size)
        :return:
        """

        q_lstm_out = self._bilstm_encode(q, q_lens)             # (batch_size, q_len, 2 * hidden_size)
        p_lstm_out = self._bilstm_encode(p, p_lens)             # (batch_size, p_len, 2 * hidden_size)

        out_q, _ = torch.max(q_lstm_out, dim=1, keepdim=False)  # (batch_size, 2 * hidden_size)
        out_p, _ = torch.max(p_lstm_out, dim=1, keepdim=False)  # (batch_size, 2 * hidden_size)

        # attention (Eq 11, 12, 13)
        weight_out_q = self.w_qm(out_q)                         # (batch_size, attention-dim)
        weight_out_p = self.w_pm(p_lstm_out)                    # (batch_size, p_len, attention_dim)
        m_aq = torch.einsum(
            "ik,ijk->ijk",
            [weight_out_q, weight_out_p]
        )                                                       # (batch_size, p_len, attention_dim)

        weights = self.w_ms(torch.tanh(m_aq))                   # (batch_size, p_len, 1)
        weights = self.att_softmax(weights)                     # (batch_size, p_len, 1)
        p_lstm_out = weights * p_lstm_out                       # (batch_size, p_len, 2 * hidden_size)
        out_p, _ = torch.max(p_lstm_out, dim=1, keepdim=False)  # (batch_size, 2 * hidden_size)

        # scores = self.cosine(out_q, out_p)

        inp = torch.cat([out_q, out_p], dim=1)                  # (batch_size, 4 * hidden_size)
        scores = torch.tanh(self.ff(inp))
        scores = torch.cat([scores, 1 - scores], dim=1)         # (batch_size, 2)

        return scores

    def init_param(self, param):
        if len(param.size()) < 2:
            init.uniform_(param)
        else:
            init.xavier_uniform_(param)


