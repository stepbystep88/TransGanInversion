import torch.nn as nn

from .transformer import TransformerEncoderBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, angle_num, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param angle_num: the number of angles
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super(BERT, self).__init__()
        self.hidden = hidden
        self.angle_num = angle_num
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, token embeddings
        self.embedding = BERTEmbedding(angle_num=angle_num, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
