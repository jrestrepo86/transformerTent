import os
import sys
import pytest
import functools
import torch
import torch.nn as nn


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oriTREET.attention import AttentionLayer, FullFixedTimeCausalConstructiveAttention
from oriTREET.transformerDecoder import DecoderLayer
from treet.attention import get_mask
from treet.transformerDecoder import TransformerBlock

seed = 42
assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-5, atol=1e-5)


def init_model_weights(model):
    for name, p in model.named_parameters():
        nn.init.ones_(p)
        # print(name)


def test_TrasnformerBlock():
    torch.manual_seed(seed)
    N = 10
    model_dim = 6
    seq_len = 4
    x = torch.rand(N, seq_len, model_dim).detach()

    parameters = {
        "heads": 3,
        "history_len": 1,
        "attn_dropout": 0.1,
        "activation": "relu",
        "trans_dropout": 0.1,
        "fordward_expansion": 4,
    }
    N, _, model_dim = x.shape
    mask = get_mask(N, seq_len, seq_len, history_len=parameters["history_len"])
    transformer_block = TransformerBlock(model_dim, **parameters)
    init_model_weights(transformer_block)

    transformer_block.eval()
    a = transformer_block(x, mask, ref_sample=False)
    b = transformer_block(x, mask, ref_sample=True)

    # TREET
    print("TREET")
    FPCA = FullFixedTimeCausalConstructiveAttention(
        mask_flag=True,
        history_len=parameters["history_len"],
        attention_dropout=parameters["attn_dropout"],
    )
    attention_TREET = AttentionLayer(FPCA, model_dim, parameters["heads"])
    init_model_weights(attention_TREET)
    transformer_block_TREET = DecoderLayer(
        [attention_TREET],
        model_dim,
        d_ff=model_dim * 4,
        dropout=parameters["trans_dropout"],
        activation=parameters["activation"],
        ff_layers=1,
    )
    init_model_weights(transformer_block_TREET)

    attention_TREET.eval()
    transformer_block_TREET.eval()
    aT = transformer_block_TREET(x, x_mask=None, drawn_y=False)
    bT = transformer_block_TREET(x, x_mask=None, drawn_y=True)

    assert_equal(a, aT)
    assert_equal(b, bT)


if __name__ == "__main__":
    test_TrasnformerBlock()
