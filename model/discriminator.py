import torch
from torch import nn

from model.embedding import BERTEmbedding
from model.transformer import TransformerEncoderBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, hidden=384, n_layers=6, attn_heads=3, dropout=0.1):
        super(Discriminator, self).__init__()
        self.emb_dropout = nn.Dropout(dropout)
        self.embedding = BERTEmbedding(angle_num=in_channels, embed_size=hidden)
        self.transform_blocks = nn.ModuleList(
            [TransformerEncoderBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1)
        )
        self.cls_token = nn.Parameter(torch.rand(hidden), requires_grad=True)

    def forward(self, x):
        x = self.embedding(x)
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_token, x), dim=1)

        for block in self.transform_blocks:
            x = block.forward(x, None)

        result = self.mlp_head(x)
        predict_label = nn.Sigmoid()(result[:, 0, :])
        return predict_label
