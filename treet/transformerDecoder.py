import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import FixedPastCausalAttention
from .embedding import Embedding


def get_activation_fn(afn):
    activation_functions = {
        "linear": lambda: lambda x: x,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
        "leaky_relu": nn.LeakyReLU,
        "threshold": nn.Threshold,
        "hardtanh": nn.Hardtanh,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "log_sigmoid": nn.LogSigmoid,
        "softplus": nn.Softplus,
        "softshrink": nn.Softshrink,
        "softsign": nn.Softsign,
        "tanhshrink": nn.Tanhshrink,
        "softmax": nn.Softmax,
        "gelu": nn.GELU,
    }

    if afn not in activation_functions:
        raise ValueError(
            f"'{afn}' is not included in activation_functions. Use below one \n {activation_functions.keys()}"
        )

    return activation_functions[afn]


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim,
        heads,
        history_len,
        attn_dropout=0.1,
        activation="relu",
        trans_dropout=0.1,
        fordward_expansion=4,
    ):
        super(TransformerBlock, self).__init__()
        # attention
        self.attention = FixedPastCausalAttention(
            model_dim, heads, history_len, attn_dropout
        )

        self.dropout = nn.Dropout(trans_dropout)

        # feed_fordward
        self.conv1 = nn.Sequential(
            nn.Conv1d(2 * model_dim, fordward_expansion * model_dim, kernel_size=1),
            get_activation_fn(activation)(),
            self.dropout,
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(fordward_expansion * model_dim, model_dim, kernel_size=1),
            self.dropout,
        )
        # Layer norms
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x, mask, ref_sample):
        input = x
        attn = self.attention(x, x, x, mask, ref_sample)
        # add and norm 1
        x = self.norm1(self.dropout(attn) + input)
        # concat
        y = torch.cat([input, x], dim=-1)
        # feed fordward
        y = self.conv1(y.transpose(-1, 1))
        y = self.conv2(y).transpose(-1, 1)
        # norm 2
        out = self.norm2(y)
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
