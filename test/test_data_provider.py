import os
import sys
import pytest
import functools
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oriTREET.data_provider import data_provider as TREET_data_provider
from oriTREET.data_provider import Dataset_Apnea
from treet.data_provider import treet_data_set, get_apnea_data, data_provider

seed = 42
assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-5, atol=1e-5)

batch_size = 10
treet_parameters = {
    "prediction_len": 10,
    "history_len": 4,
    "normalize_dataset": None,
    "source_history_len": 3,
    "last_x_zero": True,
    "batch_size": batch_size,
    "val_size": 0.2,
    "shuffle": False,
}
# TREET
TREET_args = {
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


def set_seed():
    torch.manual_seed(seed)
    np.random.seed(seed)


def test_apnea_data():

    hearth, breath = get_apnea_data("heart", "breath")
    TREET_data_set, TREET_data_loader = TREET_data_provider(TREET_args, flag)
    breathT, hearthT = TREET_data_set.data_x, TREET_data_set.data_y
    hearth = torch.Tensor(hearth)
    breath = torch.Tensor(breath)
    assert_equal(hearth.sum(), hearthT.sum())


def test_data_provider():

    set_seed()
    TREET_data_set, TREET_data_loader = TREET_data_provider(TREET_args, flag)
    set_seed()
    iter_loader = iter(TREET_data_loader)
    breathT, hearthT = next(iter_loader)

    target, source = get_apnea_data("heart", "breath")
    set_seed()
    treet_data_loader = data_provider(target, source, **treet_parameters)
    treet_training = treet_data_loader["train"]

    set_seed()
    iter_loader = iter(treet_training)
    hearth, breath = next(iter_loader)

    assert_equal(hearth, hearthT)
    assert_equal(breath, breathT)


if __name__ == "__main__":
    test_data_provider()
    test_apnea_data()
