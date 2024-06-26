import os
import sys
import pytest
import functools
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oriTREET.attention import AttentionLayer, FullFixedTimeCausalConstructiveAttention
from treet.attention import FixedPastCausalAttention as fpca
from treet.attention import get_mask

seed = 42
assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-5, atol=1e-5)


def init_model_weights(model):
    for name, w in model.named_parameters():
        # print(name)
        torch.manual_seed(seed)
        nn.init.normal_(w)


def test_attention():
    """
    Test if own implementation of attention mechanism is equal to the one in TREET
    """
    torch.manual_seed(seed)
    N = 10
    model_dim = 6
    query_len = 6
    values = torch.rand(N, query_len, model_dim).detach()
    keys = torch.rand(N, query_len, model_dim).detach()
    query = torch.rand(N, query_len, model_dim).detach()
    parameters = {"heads": 3, "history_len": 1, "dropout": 0.1}

    N, _, model_dim = values.shape
    mask = get_mask(
        N, keys.shape[1], query.shape[1], history_len=parameters["history_len"]
    )
    attention = fpca(
        model_dim=model_dim,
        history_len=parameters["history_len"],
        heads=parameters["heads"],
        attn_dropout=parameters["dropout"],
    )
    init_model_weights(attention)

    attention.eval()
    a = attention(values, keys, query, mask)
    b = attention(values, keys, query, mask, ref_sample=True)

    # TREET
    FPCA = FullFixedTimeCausalConstructiveAttention(
        mask_flag=True,
        history_len=parameters["history_len"],
        attention_dropout=parameters["dropout"],
    )
    attentionTREET = AttentionLayer(FPCA, model_dim, parameters["heads"])
    init_model_weights(attentionTREET)

    attentionTREET.eval()
    aT = attentionTREET(query, keys, values, attn_mask=None, drawn_y=False)
    bT = attentionTREET(query, keys, values, attn_mask=None, drawn_y=True)

    assert_equal(a, aT)
    assert_equal(b, bT)


if __name__ == "__main__":
    test_attention()
