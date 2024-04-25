import os
import sys
import pytest
import functools
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oriTREET.treetModel import treetModel as TREETm
from testing_parameters import oriTREETargs

seed = 42
assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-5, atol=1e-5)


def init_model_weights(model):
    for name, w in model.named_parameters():
        print(name)
        torch.manual_seed(seed)
        nn.init.normal_(w)


def run_treet_model(params):
    TREET_model = TREETm(params)
    # init_model_weights(treet_model)
    TREET_model.fit(params)
    return TREET_model.get_metrics()


if __name__ == "__main__":
    curves = run_treet_model(oriTREETargs)

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
    fig.suptitle("Curves for oriTREET", fontsize=13)
    plt.show()
