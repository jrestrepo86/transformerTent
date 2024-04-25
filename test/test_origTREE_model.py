import os
import sys
import pytest
import functools
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oriTREET.treetModel import treetModel as TREETm

args = {
    # "training"
    # Training related parameters
    "is_training": 1,  # status
    "train_epochs": 200,  # train epochs
    "batch_size": 1024,  # batch size of train input data
    "patience": 20,  # early stopping patience
    "learning_rate": 0.0001,  # optimizer learning rate
    "loss": "dv",  # loss function
    "lradj": "type1_0.95",  # adjust learning rate
    "use_amp": False,  # use automatic mixed precision training
    "optimizer": "adam",  # optimizer name, options: [adam, rmsprop]
    "n_draws": 15,  # number of draws for DV potential calculation
    "exp_clipping": "inf",  # exponential clipping for DV potential calculation
    "alpha_dv_reg": 0.0,  # alpha for DV regularization on C constant
    "num_workers": 0,  # data loader num workers
    "itr": 1,  # experiments times
    "log_interval": 5,  # training log print interval
    # "model"
    # /* Model related parameters */
    "model": "Decoder_Model",  # model name, options: [Transformer_Encoder, LSTM, Autoformer, Informer, Transformer]
    "seq_len": 0,  # input sequence length
    "label_len": 5,  # start token length. pre-prediction sequence length for the encoder
    "pred_len": 30,  # prediction sequence length
    "y_dim": 1,  # y input size - exogenous values
    "x_dim": 1,  # x input size - endogenous values
    "c_out": 1,  # output size
    "d_model": 16,  # dimension of model
    "n_heads": 1,  # num of heads
    "time_layers": 1,  # num of attention layers
    "ff_layers": 1,  # num of ff layers
    "d_ff": 32,  # dimension of fcn
    "factor": 1,  # attn factor (c hyper-parameter)
    "distil": True,  # whether to use distilling in encoder, using this argument means not using distilling
    "dropout": 0.0,  # dropout
    "embed": "fixed",  # time features encoding, options:[timeF, fixed, learned]
    "activation": "elu",  # activation - must be elu to work with NDG
    "output_attention": False,  # whether to output attention in encoder
    # "process_channel"
    # /* Process and Channel related parameters */
    "use_ndg": False,  # use NDG instead of previous created dataset.
    "process_info": {
        "type": "Apnea",
        "x": "breath",  # TE(heart->breath) < TE(breath->heart)
        "x_length": 3,  # -1 means all = label length (the last x is trimmed due to synchronization of the processes)
        "memory_cut": True,  # reset states of the model, and taking data without stride
    },
}
if __name__ == "__main__":
    m = TREETm(args)
    m.fit(args)
    train, val = m.get_curves()
