import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import FixedPastCausalAttention
from embedding import Embedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim,
        heads,
        history_len,
        attn_dropout=0.1,
        activation="relu",
        dropout=0.1,
        fordward_expansion=4,
    ):
        super(TransformerBlock, self).__init__()
        # attention
        self.attention = FixedPastCausalAttention(
            model_dim, heads, history_len, attn_dropout
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feed_fordward = nn.Sequential(
            nn.Conv1d(2 * model_dim, fordward_expansion * model_dim, kernel_size=1),
            getattr(F, activation),
            nn.Conv1d(fordward_expansion * model_dim, model_dim, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)

    def fordward(self, x, mask, ref_sample):
        input = x
        att = self.attention(x, x, x, mask, ref_sample)
        # add and norm 1
        x = self.norm1(self.dropout(att) + input)
        # concat
        y = torch.cat([input, x], dim=-1)
        # feed fordward
        fordward = self.feed_fordward(y.transpose(-1, 1))
        # norm 2
        out = self.norm2(self.dropout(fordward.transpose(-1, 1)))
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        heads,
        history_len,
        attn_dropout=0.1,
        embed_max_len=5000,
        embed_dropout=0.1,
        transf_activation="relu",
        transf_dropout=0.1,
        trans_fordward_expansion=4,
    ):
        super(Decoder, self).__init__()

        self.embedding = Embedding(input_dim, model_dim, embed_max_len, embed_dropout)
        self.transformerBlock = TransformerBlock(
            model_dim,
            heads,
            history_len,
            attn_dropout,
            transf_activation,
            transf_dropout,
            trans_fordward_expansion,
        )
        self.norm = nn.LayerNorm(self.model_dim)
        self.dense = nn.Linear(model_dim, 1)

    def fordward(self, x, mask, ref_sample):
        y = self.embedding(x)
        y = self.norm(self.transformerBlock(x, mask, ref_sample))
        out = self.dense(y)
        return out
