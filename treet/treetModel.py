import math

import torch
import torch.nn as nn

from transformerDecoder import Decoder
from treet.attention import get_mask


class treetBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        heads,
        history_len=1,
        prediction_len=1,
        attn_dropout=0.1,
        embed_max_len=5000,
        embed_dropout=0.1,
        transf_activation="relu",
        transf_dropout=0.1,
        transf_fordward_expansion=4,
    ):
        super(treetBlock, self).__init__()
        self.prediction_len = prediction_len
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

    def fordward(self, y, y_min_max, mask, x=None):
        y = torch.cat((y, x), dim=-1) if x is not None else y
        deco_out = self.decoder(y, mask, ref_sample=False)
        # random sample
        y_sampled = torch.FloatTensor(y.size).uniform_(*y_min_max).to(y.device)
        y_sampled = torch.cat((y_sampled, x), dim=-1) if x is not None else y_sampled
        sampled_deco_out = self.decoder(y_sampled, mask, ref_sample=True)

        return (
            deco_out[:, -self.prediction_len :, :],
            sampled_deco_out[:, -self.prediction_len :, :],
        )


class treetModel(nn.Module):
    def __init__(
        self,
        Y,
        X,
        y_dim=1,
        x_dim=1,
        model_dim=6,
        heads=3,
        sequence_len=0,
        label_len=5,
        prediction_len=30,
        history_len=1,
        attn_dropout=0.1,
        embed_max_len=5000,
        embed_dropout=0.1,
        transf_activation="relu",
        transf_dropout=0.1,
        trans_fordward_expansion=4,
        device=None,
    ):
        super(treetModel, self).__init__()
        # select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        torch.device(self.device)

        self.X = X
        self.Y = Y
        self.sequence_len = sequence_len
        self.label_len = label_len

        # y-model
        self.model_y = treetBlock(
            y_dim,
            model_dim,
            heads,
            history_len,
            attn_dropout,
            embed_max_len,
            embed_dropout,
            transf_activation,
            transf_dropout,
            trans_fordward_expansion,
        ).to(self.device)

        # yx-model
        self.model_yx = treetBlock(
            y_dim + x_dim,
            model_dim,
            heads,
            history_len,
            attn_dropout,
            embed_max_len,
            embed_dropout,
            transf_activation,
            transf_dropout,
            trans_fordward_expansion,
        ).to(self.device)

    def model_DV_loss(self, out, samp_out):
        """
        DV loss
        """
        t_mean = torch.mean(out)
        t_log_mean_exp = torch.logsumexp(samp_out, 0) - math.log(samp_out.shape[0])
        loss = t_mean - t_log_mean_exp
        return -1.0 * loss

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
        lr=1e-6,
        weight_decay=5e-5,
        stop_patience=100,
        stop_min_delta=0,
        val_size=0.2,
        verbose=False,
    ):
        pass
