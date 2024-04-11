import os
import sys
import pytest
import functools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oriTREET.data_provider import data_provider
from treet.data_provider import DataProvider, get_apnea_data

seed = 42
assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-5, atol=1e-5)


def test_apnea_data_provider():

    treet_parameters = {
        "prediction_len": 10,
        "history_len": 4,
        "normalize": None,
        "source_history_len": 1,
        "last_x_zero": True,
    }
    batch_size = 10

    target, source = get_apnea_data("heart", "breath")

    treet_data_set = DataProvider(target, source, **treet_parameters)
    treet_data_loader = DataLoader(
        treet_data_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # TREET
    args = {
        "model": "Decoder_Model",
        "embed": "Fixed",
        "batch_size": batch_size,
        "seq_len": 0,
        "label_len": treet_parameters["history_len"],
        "pred_len": treet_parameters["prediction_len"],
        "process_info": {
            "type": "Apnea",
            "x": "breath",  # TE(heart->breath) < TE(breath->heart)
            "x_length": treet_parameters[
                "source_history_len"
            ],  # -1 means all = label length (the last x is trimmed due to synchronization of the processes)
            "memory_cut": True,  # reset states of the model, and taking data without stride
        },
        "num_workers": 0,
        "y_dim": 1,
    }
    flag = "train"
    TREET_data_set, TREET_data_loader = data_provider(args, flag)

    torch.manual_seed(seed)
    hearth, breath = next(iter(treet_data_loader))
    torch.manual_seed(seed)
    breathT, hearthT = next(iter(TREET_data_loader))

    assert_equal(hearth, hearthT)
    assert_equal(breath, breathT)
