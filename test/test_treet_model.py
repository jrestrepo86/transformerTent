import os
import sys
import pytest
import functools
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from treet.treetModel import treetModel
from testing_parameters import treetModel_parameters
from treet.data_provider import get_apnea_data

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

    target, source = get_apnea_data("heart", "breath")
    train_params["target_signal"] = target
    train_params["source_signal"] = source
    # torch.autograd.set_detect_anomaly(True)
    treet_model.fit(**train_params)

    return treet_model.get_metrics()


if __name__ == "__main__":
    model_params = treetModel_parameters["model_params"]
    training_params = treetModel_parameters["train_params"]

    curves = run_treet_model(model_params, training_params)

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
    axs[0].plot(curves["val"]["y_loss"], "r", label="val")
    axs[0].plot(curves["train"]["y_loss"], "b", label="train")
    axs[0].set_title("y loss")

    axs[1].plot(curves["val"]["yx_loss"], "r", label="val")
    axs[1].plot(curves["train"]["yx_loss"], "b", label="train")
    axs[1].set_title("xy loss")

    axs[2].plot(curves["val"]["tent"], "r", label="val")
    axs[2].plot(curves["train"]["tent"], "b", label="train")
    axs[2].set_title("tent")
    axs[2].legend(loc="lower right")
    fig.suptitle("Curves for treet", fontsize=13)
    plt.show()
