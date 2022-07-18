import torch.nn as nn

from model.transformer import TransformerDecoderBlock
from .bert import BERT


class TransInversion(nn.Module):
    """
    基于Transformer的反演网络
    """
    def __init__(self, angle_num, hidden=768, n_layers=12, n_decoder_layers=12, attn_heads=12, dropout=0.1):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.encoder = BERT(angle_num, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads, dropout=dropout)
        self.d_decoder = Decoder(hidden, angle_num, attn_heads=attn_heads, dropout=dropout, n_layers=n_decoder_layers)
        self.well_decoder = Decoder(hidden, 3, attn_heads=attn_heads, dropout=dropout, n_layers=n_decoder_layers)

    def forward(self, d, init_data, mask):
        d_encoding = self.encoder(d, mask)

        d_predict = self.d_decoder(d_encoding, d_encoding)
        well_predict = self.well_decoder(d_encoding, d_encoding) + init_data

        return d_predict, well_predict


class Decoder(nn.Module):
    """
    将地震数据的encoding反推为特定目标
    """
    def __init__(self, hidden, out_channel, attn_heads=3, dropout=0.1, n_layers=2):
        """
        :param hidden: output size of BERT model
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerDecoderBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.linear = nn.Sequential(nn.Conv1d(hidden, out_channel, 1, 1, 0))

    def forward(self, x, y):
        for block in self.blocks:
            x = block(x, y)

        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
