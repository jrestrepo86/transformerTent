import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .attention import get_mask
from .data_provider import data_provider, toColVector, treet_data_set
from .transformerDecoder import Decoder


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
        yt = torch.cat((y, x), dim=-1) if x is not None else y
        deco_out = self.decoder(yt, mask, ref_sample=False)

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
        target_signal,
        source_signal,
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

        self.prediction_len = prediction_len
        self.history_len = history_len
        self.target = torch.tensor(toColVector(target_signal), dtype=torch.float).to(
            self.device
        )
        self.source = torch.tensor(toColVector(source_signal), dtype=torch.float).to(
            self.device
        )
        self.target_dim = target_signal.shape[1]
        self.source_dim = source_signal.shape[1]

        # y-model
        self.model_y = treetBlock(
            input_dim=self.target_dim,
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
            input_dim=self.target_dim + self.source_dim,
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
        t_log_mean_exp = torch.log(torch.sum(torch.exp(samp_out))) - math.log(
            samp_out.shape[0] * samp_out.shape[1]
        )
        loss = t_mean - t_log_mean_exp
        return -loss

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
        lr=1e-6,
        weight_decay=5e-5,
        train_size=0.8,
        val_size=0.2,
        normalize_dataset=None,
        calc_tent=True,
        source_history_len=None,
        verbose=False,
    ):

        self.data_loaders = data_provider(
            target=self.target,
            source=self.source,
            prediction_len=self.prediction_len,
            history_len=self.history_len,
            normalize_dataset=normalize_dataset,
            source_history_len=source_history_len,
            last_x_zero=calc_tent,
            train_size=train_size,
            val_size=val_size,
            batch_size=batch_size,
        )

        # set optimizer
        opt_y = torch.optim.Adam(
            self.model_y.parameters(), lr=lr, weight_decay=weight_decay
        )
        opt_yx = torch.optim.Adam(
            self.model_yx.parameters(), lr=lr, weight_decay=weight_decay
        )

        # data data_provider
        y_min_max = torch.Tensor(
            [
                torch.minimum(self.target.min(), self.source.min()),
                torch.maximum(self.target.max(), self.source.max()),
            ]
        )

        # set full fixed attention mask
        sequence_len = self.prediction_len + self.history_len

        # training
        val_loss_epoch = []
        val_y_loss_epoch = []
        val_yx_loss_epoch = []
        tent_epoch = []
        for _ in tqdm(range(max_epochs), disable=not verbose):
            self.model_state("train")
            with torch.set_grad_enabled(True):
                for _, (target_b, source_b) in enumerate(self.data_loaders["train"]):
                    mask = get_mask(
                        target_b.shape[0], sequence_len, sequence_len, self.history_len
                    ).to(self.device)
                    opt_y.zero_grad()
                    opt_yx.zero_grad()
                    y, samp_y = self.model_y(target_b, y_min_max, mask, x=None)
                    yx, samp_yx = self.model_yx(target_b, y_min_max, mask, x=source_b)
                    loss_y = self.model_DV_loss(y, samp_y)
                    loss_yx = self.model_DV_loss(yx, samp_yx)
                    loss = loss_y + loss_yx
                    loss.backward()
                    opt_y.step()
                    opt_yx.step()

            # validation
            self.model_state("eval")
            with torch.no_grad():
                ml, my, myx, mtent = [], [], [], []
                for _, (target_b, source_b) in enumerate(self.data_loaders["val"]):
                    mask = get_mask(
                        target_b.shape[0], sequence_len, sequence_len, self.history_len
                    ).to(self.device)
                    y, samp_y = self.model_y(target_b, y_min_max, mask, x=None)
                    yx, samp_yx = self.model_yx(target_b, y_min_max, mask, x=source_b)
                    loss_y = self.model_DV_loss(y, samp_y)
                    loss_yx = self.model_DV_loss(yx, samp_yx)
                    loss = loss_y + loss_yx
                    tent = loss_y - loss_yx
                    ml.append(loss.item())
                    my.append(-loss_y.item())
                    myx.append(-loss_yx.item())
                    mtent.append(tent.item())

                val_loss_epoch.append(np.array(ml).mean())
                val_y_loss_epoch.append(np.array(my).mean())
                val_yx_loss_epoch.append(np.array(myx).mean())
                tent_epoch.append(np.array(mtent).mean())

        self.val_loss_epoch = np.array(val_loss_epoch)
        self.val_y_loss_epoch = np.array(val_y_loss_epoch)
        self.val__yx_loss_epoch = np.array(val_yx_loss_epoch)
        self.tent_epoch = np.array(tent_epoch)

    def get_curves(self):
        return (
            self.val_loss_epoch,
            self.val_y_loss_epoch,
            self.val__yx_loss_epoch,
            self.tent_epoch,
        )
