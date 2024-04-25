import torch.nn as nn
import torch
from torch import optim
import numpy as np

from .decoder_model import Model
from .data_provider import data_provider
from .metrics import DV_Loss


class treetModel(nn.Module):
    def __init__(self, args, device=None):
        super(treetModel, self).__init__()

        self.args = args
        # select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        torch.device(self.device)

        self.args["x_in"] = False
        self.model_y = Model(self.args).float().to(self.device)
        self.args["x_in"] = True
        self.model_yx = Model(self.args).float().to(self.device)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def reset_states(self, mode="train"):
        for model in [self.model_y, self.model_yx]:
            if mode == "train":
                model.train()
            elif mode == "eval":
                model.eval()
            if hasattr(model, "erase_states"):
                model.erase_states()
        if hasattr(self, "channel"):
            self.channel.erase_states()

    def fit(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        opt_y = optim.Adam(self.model_y.parameters(), lr=self.args["learning_rate"])
        opt_yx = optim.Adam(self.model_yx.parameters(), lr=self.args["learning_rate"])

        criterion = DV_Loss(self.args["exp_clipping"], self.args["alpha_dv_reg"])

        val_metrics = {"y": [], "yx": [], "tent": []}
        train_metrics = {"y": [], "yx": [], "tent": []}
        for epoch in range(self.args["train_epochs"]):
            iter_count = 0

            self.reset_states(mode="train")
            with torch.set_grad_enabled(True):
                tm_y, tm_yx = [], []
                for _, (batch_x, batch_y) in enumerate(train_loader):
                    if self.args["process_info"]["memory_cut"]:
                        self.reset_states()
                    iter_count += 1
                    opt_y.zero_grad()
                    opt_yx.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    # encoder - decoder
                    outputs_y = self.model_y(batch_y, y_tilde=train_data.min_max)
                    outputs_yx = self.model_yx(
                        batch_y, y_tilde=train_data.min_max, x=batch_x
                    )
                    loss_y = criterion(outputs_y[:2])
                    loss_yx = criterion(outputs_yx[:2])
                    loss = loss_yx + loss_y
                    tm_y.append(-loss_y.item())
                    tm_yx.append(-loss_yx.item())
                    loss.backward()
                    opt_y.step()
                    opt_yx.step()
            train_metrics["y"].append(np.array(tm_y).mean())
            train_metrics["yx"].append(np.array(tm_yx).mean())
            train_metrics["tent"].append(np.array(tm_yx).mean() - np.array(tm_y).mean())

            self.reset_states(mode="eval")
            with torch.no_grad():
                vm_y, vm_yx = [], []
                for i, (batch_x, batch_y) in enumerate(vali_loader):
                    if self.args["process_info"]["memory_cut"]:
                        self.reset_states()

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    # encoder - decoder
                    outputs_y = self.model_y(batch_y, y_tilde=vali_data.min_max)
                    outputs_yx = self.model_yx(
                        batch_y, y_tilde=vali_data.min_max, x=batch_x
                    )
                    loss_y = criterion(outputs_y[:2])
                    loss_yx = criterion(outputs_yx[:2])
                    vm_y.append(-loss_y.item())
                    vm_yx.append(-loss_yx.item())
            val_metrics["y"].append(np.array(vm_y).mean())
            val_metrics["yx"].append(np.array(vm_yx).mean())
            val_metrics["tent"].append(np.array(tm_yx).mean() - np.array(tm_y).mean())

        for k in train_metrics.keys():
            train_metrics[k] = np.array(train_metrics[k])
            val_metrics[k] = np.array(val_metrics[k])
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

    def get_curves(self):
        return self.val_metrics, self.train_metrics


if __name__ == "__main__":
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

    m = treetModel(args)
    m.fit(args)
