import os
import sys
import pytest
import functools
import torch
import torch.nn as nn


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oriTREET.decoder_model import Model as TREETm

from treet.attention import get_mask
from treet.treetModel import treetBlock

seed = 42
assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-5, atol=1e-5)


def init_model_weights(model):
    for name, w in model.named_parameters():
        # print(name)
        torch.manual_seed(seed)
        nn.init.normal_(w)


def run_TREET_model(params, y, y_min_max):
    print("TREET Model")
    args = {
        "y_dim": params["input_dim"],
        "x_dim": 1,
        "d_model": params["model_dim"],
        "label_len": params["history_len"],
        "pred_len": params["prediction_len"],
        "n_heads": params["heads"],
        "d_ff": params["transf_fordward_expansion"] * params["model_dim"],
        "dropout": params["attn_dropout"],
        "activation": params["transf_activation"],
        "n_draws": params["treet_block_randsamp_draws"],
        "time_layers": 1,
        "factor": 5,
        "ff_layers": 1,
        "x_in": False,
        "c_out": 1,
        "batch_size": 3,
        "seq_len": 0,
        "output_attention": False,
        "process_info": {
            "type": "Apnea",
            "x": "breath",  # TE(heart->breath) < TE(breath->heart)
            "x_length": 3,  # -1 means all = label length (the last x is trimmed due to synchronization of the processes)
            "memory_cut": True,  # reset states of the model, and taking data without stride
        },
        "num_workers": 0,
        "model": "Decoder_Model",
        "embed": "Fixed",
    }
    TREET_model = TREETm(configs=args)
    init_model_weights(TREET_model)

    torch.manual_seed(seed)
    TREET_model.eval()
    a, b = TREET_model(y, y_min_max)
    return a, b


def run_treet_block(params, y, y_min_max):
    print("treet block")
    batch_size = y.shape[0]
    history_len = params["history_len"]
    sequence_len = history_len + params["prediction_len"]

    mask = get_mask(batch_size, sequence_len, sequence_len, history_len)

    treet_block_model = treetBlock(**params)
    init_model_weights(treet_block_model)

    torch.manual_seed(seed)
    treet_block_model.eval()
    a, b = treet_block_model(y, y_min_max, mask, x=None)
    return a, b


def test_treet_block():
    torch.manual_seed(seed)
    input_dim = 3
    model_dim = 6
    history_len = 1
    prediction_len = 8
    batch_size = 10
    sequence_len = prediction_len + history_len
    y = torch.rand(batch_size, sequence_len, input_dim).detach()
    y_min_max = torch.Tensor([0, 1]).detach()
    parameters = {
        "input_dim": input_dim,
        "model_dim": model_dim,
        "heads": 2,
        "history_len": history_len,
        "prediction_len": prediction_len,
        "attn_dropout": 0.0,
        "embed_max_len": 5000,
        "embed_dropout": 0.0,
        "transf_activation": "relu",
        "transf_dropout": 0.0,
        "transf_fordward_expansion": 4,
        "treet_block_randsamp_draws": 15,
    }
    aT, bT = run_TREET_model(parameters, y, y_min_max)
    a, b = run_treet_block(parameters, y, y_min_max)
    assert_equal(a, aT)
    assert_equal(b, bT)
