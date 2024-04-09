import os
import sys
import pytest
import functools
import torch
import torch.nn as nn


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oriTREET.embed import DataEmbedding_wo_temp
from treet.embedding import Embedding

seed = 42
assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-5, atol=1e-5)


def init_model_weights(model):
    for name, p in model.named_parameters():
        nn.init.ones_(p)
        # print(name)


def test_embedding():
    """
    Test if own implementation of embedding mechanism is equal to the one in TREET
    """
    torch.manual_seed(seed)
    N = 10
    input_dim = 2
    query_len = 4
    model_dim = 6
    input = torch.rand(N, query_len, input_dim).detach()
    parameters = {"max_len": 5000, "dropout": 0.1}

    embedding = Embedding(
        input_dim,
        model_dim,
        max_len=parameters["max_len"],
        embd_dropout=parameters["dropout"],
    )
    init_model_weights(embedding)

    # TREET
    embeddingTREET = DataEmbedding_wo_temp(
        input_dim, model_dim, dropout=parameters["dropout"]
    )
    init_model_weights(embeddingTREET)

    embedding.eval()
    a = embedding(input)
    embeddingTREET.eval()
    b = embeddingTREET(input)
    assert_equal(a, b)
