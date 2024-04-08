import os
import sys
import pytest
import functools
import torch
from torch.nn.modules import dropout


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oriTREET.embed import DataEmbedding_wo_temp
from treet.embedding import Embedding

seed = 42
assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-5, atol=1e-5)


def test_embedding():
    """
    Test if own implementation of embedding mechanism is equal to the one in TREET
    """
    N = 4
    input_dim = 4
    model_dim = 6
    input = torch.rand(N, input_dim).detach()
    parameters = {"max_len": 5000, "dropout": 0.1}

    torch.manual_seed(seed)
    emb1 = Embedding(
        input_dim,
        model_dim,
        max_len=parameters["max_len"],
        embd_dropout=parameters["dropout"],
    )

    # TREET
    torch.manual_seed(seed)
    emb2 = DataEmbedding_wo_temp(input_dim, model_dim, dropout=parameters["dropout"])

    emb1.eval()
    a = emb1(input)
    emb2.eval()
    b = emb2(input)
    assert_equal(a, b)
