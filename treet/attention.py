import torch
import torch.nn as nn


def get_mask(N, key_len, query_len, history_len):
    """
    Compute the Toeplitz-like band matrix mask
    """
    m1 = torch.tril(torch.ones((query_len, key_len), dtype=torch.bool))
    m2 = torch.tril(
        torch.ones((query_len, key_len), dtype=torch.bool),
        diagonal=-(history_len + 1),
    )
    mask = torch.logical_not(torch.bitwise_xor(m1, m2))
    mask = mask.expand(N, 1, query_len, key_len)
    return mask


class FixedPastCausalAttention(nn.Module):
    def __init__(self, model_dim, heads, history_len, mask):
        super(FixedPastCausalAttention, self).__init__()
        self.model_dim = model_dim
        self.heads = heads
        self.history_len = history_len
        self.mask = mask
        self.head_dim = model_dim // heads
        self.scale = model_dim ** (1 / 2)

        assert self.head_dim * heads == model_dim, "Model dim needs to be div by heads"

        self.values_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(model_dim, model_dim)

        self.origin_values = []
        self.origin_keys = []
        self.origin_query = []

    def fpca_attention(self, values, keys, query):
        """
        full past causal attention energy
        Inputs shape (N, query/keys_len, heads, heads_dim)
        energy shape (N, heads, query_len, keys_len)
        attention shape (N, query_len, heads, heads_dim)
        """
        # energy
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])
        # masking
        energy = energy.masked_fill(self.mask, float(-1e20))
        # scores
        scores = torch.softmax(energy / self.scale, dim=-1)
        # attention shape (N, query_len, heads, heads_dim)
        attention = torch.einsum("nhql,nlhd->nlhd", [scores, values])
        return attention

    def mfpca_attention(self, values, keys, query):
        """
        modified full past causal attention energy
        Inputs shape (N, query/keys_len, heads, heads_dim)
        energy shape (N, heads, query_len, keys_len)
        attention shape (N, query_len, heads, heads_dim)
        """
        origin_keys = self.origin_keys.clone()
        origin_values = self.origin_values.clone()
        # energy
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, origin_keys])
        samp_energy = torch.einsum("nlhd,nlhd->nhl", [query, keys])
        energy.diagonal(offset=0, dim1=2, dim2=3).copy_(samp_energy)
        # masking
        energy = energy.masked_fill(self.mask, float(-1e20))
        # scores
        scores = torch.softmax(energy / self.scale, dim=-1)
        scores_diag = torch.diag_embed(
            scores.diagonal(offset=0, dim1=2, dim2=3), offset=0, dim1=2, dim2=3
        )
        torch.diagonal(scores, offset=0, dim1=2, dim2=3).zero_()
        # attention
        origin_attention = torch.einsum("nhql,nlhd->nlhd", [scores, origin_values])
        samp_attention = torch.einsum("nhql,nlhd->nlhd", [scores_diag, values])
        return samp_attention + origin_attention

    def forward(self, values, keys, query, ref_sample=False):
        """
        input shape (N, query_len, model_dim)(samples, sequence lenght, model dimension)
        """
        N = query.shape[0]
        values_len, keys_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split model_dim into heads
        values = values.reshape(N, values_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        # Calculate projections
        values = self.values_proj(values)
        keys = self.keys_proj(keys)
        query = self.query_proj(query)

        # get attention
        if ref_sample:
            attention = self.mfpca_attention(values, keys, query)
        else:
            self.origin_values = values
            self.origin_keys = keys
            self.origin_query = query
            attention = self.fpca_attention(values, keys, query)

        # concat heads, out shape (N, query_len, model_dim)
        out = attention.reshape(N, query_len, self.model_dim)
        # fully conected layer
        out = self.fc_out(out)

        return out
