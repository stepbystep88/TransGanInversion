import torch.nn as nn

from model.embedding import BERTEmbedding
from model.transformer import TransformerDecoderBlock
from .bert import BERT


class Decoder(nn.Module):
    def __init__(self, hidden, attn_heads=3, n_layers=12, dropout=0.1):
        super().__init__()
        # embedding for BERT, sum of positional, token embeddings
        self.embedding = BERTEmbedding(angle_num=3, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.decoder_blocks = nn.ModuleList(
            [TransformerDecoderBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, y):
        x = self.embedding(x)

        for transformer in self.decoder_blocks:
            x = transformer.forward(x, y)
        return x


class TransInversion(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, n_layers=12, dropout=0.1):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.decoder = Decoder(self.bert.hidden, n_layers=n_layers, dropout=dropout)
        self.head_d = DecodeHead(self.bert.hidden, self.bert.angle_num)
        self.head_well_data = DecodeHead(self.bert.hidden, 3)   # vp, vs, rho 三个属性

    def forward(self, d, init_data):
        d_encoding = self.bert(d)
        init_well_encoding = self.decoder(init_data, d_encoding)

        d_new = self.head_d(d_encoding)
        well_new = self.head_well_data(init_well_encoding)
        return d_new, well_new


class DecodeHead(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, out_channel):
        """
        :param hidden: output size of BERT model
        """
        super().__init__()
        self.linear1 = nn.Linear(hidden, hidden)
        self.activate = nn.ReLU()
        self.linear2 = nn.Linear(hidden, out_channel)

    def forward(self, x):
        return self.linear2(self.activate(self.linear1(x)))
