import torch
import torch.nn as nn
import numpy as np

from model.transformer import TransformerEncoderBlock
from model.unet import UNet
from model.utils import AdaInLinear
from .bert import BERT


class TransInversion(nn.Module):
    """
    基于Transformer的反演网络
    """
    def __init__(self, angle_num, seq_len, hidden=768, n_layers=12, n_decoder_layers=12, attn_heads=12, dropout=0.1):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.encoder = BERT(angle_num, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads, dropout=dropout)
        self.d_decoder = SeismicDecoder(hidden, angle_num)
        self.well_decoder = WellDecoder(hidden, 3, seq_len,
                                        attn_heads=attn_heads, dropout=dropout, n_layers=n_decoder_layers)
        self.u_net = UNet(enc_chs=(3, 16, 32, 64, 128, 256, 512), dec_chs=(512, 256, 128, 64, 32, 16), out_channel=3)

    def forward(self, d, init_data):
        d_encoding = self.encoder(d, None)

        d_predict = self.d_decoder(d_encoding)
        well_predict = self.well_decoder(d_encoding, init_data)
        well_predict_final = self.u_net(well_predict)

        return d_predict, well_predict, well_predict_final


class SeismicDecoder(nn.Module):
    """
    将地震数据的encoding反推为特定目标
    """
    def __init__(self, hidden, out_channel):
        """
        :param hidden: output size of BERT model
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, out_channel)
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class WellDecoder(nn.Module):
    def __init__(self, hidden, out_channel, seq_len, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.linear1 = nn.Sequential(
            nn.Linear(hidden, out_channel)
        )
        self.adain_linear = AdaInLinear(hidden, out_channel)

        mask = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(0, i+1):
                mask[i, j] = 1

        self.mask = nn.Parameter(torch.Tensor(mask).unsqueeze(0).repeat(attn_heads, 1, 1))

    def forward(self, x, z):
        for block in self.blocks:
            x = block(x, self.mask)

        x = self.linear1(x)
        x = x + z
        x = self.adain_linear(x, z)

        return x
