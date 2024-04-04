import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oriTREET.attention import (AttentionLayer,
                                FullFixedTimeCausalConstructiveAttention)
from treet.attention import FixedPastCausalAttention as fpca
from treet.attention import get_mask


def test_treet(values, keys, query, parameters):
    N, _, model_dim = values.shape

    mask = get_mask(
        N, keys.shape[1], query.shape[1], history_len=parameters["history_len"]
    )

    attention = fpca(
        model_dim=model_dim,
        history_len=parameters["history_len"],
        mask=mask,
        heads=parameters["heads"],
    )

    a = attention(values, keys, query)
    b = attention(values, keys, query, ref_sample=True)
    return a, b


def test_TREET(values, keys, query, parameters):
    _, _, model_dim = values.shape
    FPCA = FullFixedTimeCausalConstructiveAttention(
        mask_flag=True,
        history_len=parameters["history_len"],
        attention_dropout=0.0,
    )
    attention = AttentionLayer(FPCA, model_dim, parameters["heads"])
    a = attention(query, keys, values, attn_mask=None, drawn_y=False)
    b = attention(query, keys, values, attn_mask=None, drawn_y=True)
    return a, b


if __name__ == "__main__":
    N = 2
    model_dim = 4
    query_len = 3
    values1 = torch.rand(N, query_len, model_dim)
    keys1 = torch.rand(N, query_len, model_dim)
    query1 = torch.rand(N, query_len, model_dim)
    values2 = values1.detach().clone()
    keys2 = keys1.detach().clone()
    query2 = query1.detach().clone()

    # values = torch.arange(model_dim, dtype=torch.float).expand(
    #     [N, query_len, model_dim]
    # )
    # keys = torch.arange(N * model_dim * query_len, dtype=torch.float).reshape(
    #     N, query_len, model_dim
    # )
    # query = torch.ones_like(keys)
    parameters = {"heads": 2, "history_len": 1}
    at, bt = test_treet(values2, keys2, query2, parameters)
    aT, bT = test_TREET(values1, keys1, query1, parameters)
    # print(at)
    # print(aT)
    print(torch.einsum("nld->", bT - bt))
    print(torch.einsum("nld->", aT - at))
