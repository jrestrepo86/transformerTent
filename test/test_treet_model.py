import os
import sys
import pytest
import functools
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oriTREET.decoder_model import Model as TREETm

from treet.attention import get_mask
from treet.treetModel import treetModel

seed = 42
assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-5, atol=1e-5)


def init_model_weights(model):
    for name, w in model.named_parameters():
        print(name)
        torch.manual_seed(seed)
        nn.init.normal_(w)


def run_treet_model(model_params, train_params):
    print("treet model")

    treet_model = treetModel(**model_params)
    # init_model_weights(treet_model)

    torch.manual_seed(seed)
    treet_model.eval()
    treet_model.fit(**train_params)
    val, tent = treet_model.get_curves()
    return val, tent


def test_treet_model():
    # torch.manual_seed(seed)
    np.random.seed(seed=seed)
    N = 10000
    model_dim = 4
    history_len = 1
    prediction_len = 9
    batch_size = 128
    target = np.random.normal(0, 1, (N, 1))
    source = np.random.normal(0, 1, (N, 1))

    model_params = {
        "target_signal": target,
        "source_signal": source,
        "model_dim": model_dim,
        "heads": 2,
        "history_len": history_len,
        "prediction_len": prediction_len,
        "attn_dropout": 0.1,
        "embed_max_len": 5000,
        "embed_dropout": 0.1,
        "transf_activation": "relu",
        "transf_dropout": 0.1,
        "transf_fordward_expansion": 4,
        "treet_block_randsamp_draws": 15,
    }

    train_params = {
        "batch_size": batch_size,
        "max_epochs": 150,
        "lr": 1e-5,
        "weight_decay": 5e-5,
        "val_size": 0.2,
        "test_set": False,
        "normalize_dataset": None,
        "calc_tent": True,
        "source_history": None,
        "verbose": True,
    }

    # aT, bT = run_TREET_model(parameters, y, y_min_max)
    # torch.autograd.set_detect_anomaly(True)
    val, tent = run_treet_model(model_params, train_params)
    plt.plot(val)
    plt.plot(tent)
    plt.show()
    # assert_equal(a, aT)
    # assert_equal(b, bT)


if __name__ == "__main__":
    test_treet_model()
