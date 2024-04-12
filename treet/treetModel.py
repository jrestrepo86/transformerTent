import math

import torch
import torch.nn as nn
import numpy as np

from .transformerDecoder import Decoder
from .attention import get_mask
from .data_provider import treet_data_set
from torch.utils.data import DataLoader, Subset


class treetBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        heads,
        history_len=1,
        prediction_len=8,
        attn_dropout=0.1,
        embed_max_len=5000,
        embed_dropout=0.1,
        transf_activation="relu",
        transf_dropout=0.1,
        transf_fordward_expansion=4,
        treet_block_randsamp_draws=15,
    ):
        super(treetBlock, self).__init__()
        self.prediction_len = prediction_len
        self.treet_block_randsamp_draws = treet_block_randsamp_draws
        self.decoder = Decoder(
            input_dim,
            model_dim,
            heads,
            history_len,
            attn_dropout,
            embed_max_len,
            embed_dropout,
            transf_activation,
            transf_dropout,
            transf_fordward_expansion,
        )

    def forward(self, y, y_min_max, mask, x=None):
        y = torch.cat((y, x), dim=-1) if x is not None else y
        deco_out = self.decoder(y, mask, ref_sample=False)

        # random samples
        temp_array = []
        for _ in range(self.treet_block_randsamp_draws):
            y_sampled = torch.FloatTensor(y.size()).uniform_(*y_min_max).to(y.device)
            y_sampled = (
                torch.cat((y_sampled, x), dim=-1) if x is not None else y_sampled
            )
            sampled_deco_out = self.decoder(y_sampled, mask, ref_sample=True)
            temp_array.append(sampled_deco_out)
        sampled_deco_out = torch.stack(temp_array)

        return (
            deco_out[:, -self.prediction_len :, :],
            sampled_deco_out[..., -self.prediction_len :, :],
        )


class treetModel(nn.Module):
    def __init__(
        self,
        y_dim,
        x_dim,
        model_dim=6,
        heads=3,
        history_len=1,
        prediction_len=8,
        attn_dropout=0.1,
        embed_max_len=5000,
        embed_dropout=0.1,
        transf_activation="relu",
        transf_dropout=0.1,
        transf_fordward_expansion=4,
        treet_block_randsamp_draws=15,
        device=None,
    ):
        super(treetModel, self).__init__()
        # select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        torch.device(self.device)

        # y-model
        self.model_y = treetBlock(
            input_dim=y_dim,
            model_dim=model_dim,
            heads=heads,
            history_len=history_len,
            prediction_len=prediction_len,
            attn_dropout=attn_dropout,
            embed_max_len=embed_max_len,
            embed_dropout=embed_dropout,
            transf_activation=transf_activation,
            transf_dropout=transf_dropout,
            transf_fordward_expansion=transf_fordward_expansion,
            treet_block_randsamp_draws=treet_block_randsamp_draws,
        ).to(self.device)

        # yx-model
        self.model_yx = treetBlock(
            input_dim=y_dim + x_dim,
            model_dim=model_dim,
            heads=heads,
            history_len=history_len,
            prediction_len=prediction_len,
            attn_dropout=attn_dropout,
            embed_max_len=embed_max_len,
            embed_dropout=embed_dropout,
            transf_activation=transf_activation,
            transf_dropout=transf_dropout,
            transf_fordward_expansion=transf_fordward_expansion,
            treet_block_randsamp_draws=treet_block_randsamp_draws,
        ).to(self.device)

    def model_state(self, state="train"):
        if state == "train":
            self.model_y.train()
            self.model_yx.train()
        if state == "eval":
            self.model_y.eval()
            self.model_yx.eval()

    def model_DV_loss(self, out, samp_out):
        """
        DV loss
        """
        t_mean = torch.mean(out)
        t_log_mean_exp = torch.logsumexp(samp_out, 0) - math.log(samp_out.shape[0])
        loss = t_mean - t_log_mean_exp
        return -loss

    def data_provider(self, target, source):

        train_dataset = treet_data_set(
            target,
            source,
            prediction_len=self.prediction_len,
            history_len=self.history_len,
            normalize=self.normalize_dataset,
            source_history_len=self.source_history,
            last_x_zero=self.calc_tent,
        )
        val_dataset = treet_data_set(
            target,
            source,
            prediction_len=self.prediction_len,
            history_len=self.history_len,
            normalize=self.normalize_dataset,
            source_history_len=self.source_history,
            last_x_zero=self.calc_tent,
        )
        self.test_dataset = []
        if self.test_set:
            self.test_dataset = treet_data_set(
                target,
                source,
                prediction_len=self.prediction_len,
                history_len=self.history_len,
                normalize=self.normalize_dataset,
                source_history_len=self.source_history,
                last_x_zero=self.calc_tent,
            )

        # split in train val
        n = train_dataset.data_len
        val_size = int(n * self.val_size)
        inds = torch.randperm(n, dtype=torch.int)
        (val_idx, train_idx) = (inds[:val_size], inds[val_size:])
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(val_dataset, val_idx)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

    def fit(
        self,
        target_signal,
        source_signal,
        batch_size=64,
        max_epochs=2000,
        lr=1e-6,
        weight_decay=5e-5,
        val_size=0.2,
        test_set=False,
        normalize_dataset=None,
        calc_tent=True,
        source_history=None,
        verbose=False,
    ):

        self.batch_size = batch_size
        self.val_size = val_size
        self.test_set = test_set
        self.normalize_dataset = normalize_dataset
        self.calc_tent = calc_tent
        self.source_history = source_history

        # set optimizer
        opt_y = torch.optim.Adam(
            self.model_y.parameters(), lr=lr, weight_decay=weight_decay
        )
        opt_yx = torch.optim.Adam(
            self.model_yx.parameters(), lr=lr, weight_decay=weight_decay
        )

        # data data_provider
        self.data_provider(target_signal, source_signal)
        y_min_max = torch.Tensor(
            [
                min(target_signal.min(), source_signal.min()),
                max(target_signal.max(), source_signal.max()),
            ]
        )

        # set full fixed attention mask
        sequence_len = self.prediction_len + self.history_len
        mask = get_mask(batch_size, sequence_len, sequence_len, self.history_len)

        # trainin
        val_loss_epoch = []
        for _ in range(max_epochs):
            self.model_state("train")
            for i, (target_b, source_b) in enumerate(self.train_loader):
                y, samp_y = self.model_y(target_b, y_min_max, mask, x=None)
                yx, samp_yx = self.model_y(target_b, y_min_max, mask, x=source_b)
                loss_y = self.model_DV_loss(y, samp_y)
                loss_yx = self.model_DV_loss(yx, samp_yx)
                loss = loss_y + loss_yx
                loss.backward()
                opt_y.step()
                opt_yx.step()

            # validation
            self.model_state("eval")
            with torch.no_grad():
                for i, (target_b, source_b) in enumerate(self.val_loader):
                    y, samp_y = self.model_y(target_b, y_min_max, mask, x=None)
                    yx, samp_yx = self.model_y(target_b, y_min_max, mask, x=source_b)
                    loss_y = self.model_DV_loss(y, samp_y)
                    loss_yx = self.model_DV_loss(yx, samp_yx)
                    loss = loss_y + loss_yx
                    val_loss_epoch.append(loss.mean().item())

        self.val_loss_epoch = np.array(val_loss_epoch)

    def get_curves(self):
        return self.val_loss_epoch
