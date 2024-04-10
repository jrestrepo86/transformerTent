import os
import sys
import pytest
import functools
import torch
import torch.nn as nn


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oriTREET.attention import AttentionLayer, FullFixedTimeCausalConstructiveAttention
from oriTREET.transformerDecoder import DecoderLayer
from oriTREET.transformerDecoder import Decoder as TREETDecoder
from treet.attention import get_mask
from treet.transformerDecoder import Decoder as treetDecoder

seed = 42
assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-5, atol=1e-5)


def init_model_weights(model):
    for name, w in model.named_parameters():
        # print(name)
        torch.manual_seed(seed)
        nn.init.normal_(w)


def test_Decoder():
    torch.manual_seed(seed)
    N = 10
    input_dim = 3
    model_dim = 6
    seq_len = 4
    x = torch.rand(N, seq_len, input_dim).detach()
    parameters = {
        "heads": 3,
        "history_len": 1,
        "attn_dropout": 0.0,
        "embed_max_len": 5000,
        "embed_dropout": 0.0,
        "transf_activation": "relu",
        "transf_dropout": 0.0,
        "transf_fordward_expansion": 4,
    }
    print("treet model")
    mask = get_mask(N, seq_len, seq_len, history_len=parameters["history_len"])
    treet_decoder = treetDecoder(input_dim, model_dim, **parameters)
    init_model_weights(treet_decoder)

    treet_decoder.eval()
    a = treet_decoder(x, mask, ref_sample=False)
    b = treet_decoder(x, mask, ref_sample=True)

    # TREET
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
        dropout=parameters["transf_dropout"],
        activation=parameters["transf_activation"],
        ff_layers=1,
    )
    init_model_weights(transformer_block_TREET)

    decoder_TREET = TREETDecoder(
        [transformer_block_TREET],
        norm_layer=torch.nn.LayerNorm(model_dim),
        projection=nn.Linear(model_dim, 1, bias=True),
    )
    init_model_weights(decoder_TREET)

    attention_TREET.eval()
    transformer_block_TREET.eval()
    decoder_TREET.eval()

    aT = decoder_TREET(x, x_mask=None, drawn_y=False)
    bT = decoder_TREET(x, x_mask=None, drawn_y=True)


if __name__ == "__main__":
    test_Decoder()
