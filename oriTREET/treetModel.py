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

        val_metrics = {"y_loss": [], "yx_loss": [], "tent": []}
        train_metrics = {"y_loss": [], "yx_loss": [], "tent": []}
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
            train_metrics["y_loss"].append(np.array(tm_y).mean())
            train_metrics["yx_loss"].append(np.array(tm_yx).mean())
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
            val_metrics["y_loss"].append(np.array(vm_y).mean())
            val_metrics["yx_loss"].append(np.array(vm_yx).mean())
            val_metrics["tent"].append(np.array(tm_yx).mean() - np.array(tm_y).mean())

        for k in train_metrics.keys():
            train_metrics[k] = np.array(train_metrics[k])
            val_metrics[k] = np.array(val_metrics[k])
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

    def get_metrics(self):
        metrics = {"val": self.val_metrics, "train": self.train_metrics}
        return metrics
