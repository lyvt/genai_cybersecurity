import torch
import torch.nn as nn
from uer.layers.transformer import TransformerLayer
from uer.layers.relative_position_embedding import RelativePositionEmbedding


class TransformerEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.transformer = TransformerLayer(args)
        self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num, num_buckets=args.relative_attention_buckets_num)

    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, _ = emb.size()
        mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        hidden = emb
        position_bias = self.relative_pos_emb(hidden, hidden)
        for i in range(self.layers_num):
            hidden = self.transformer(hidden, mask, position_bias=position_bias)
        return hidden
