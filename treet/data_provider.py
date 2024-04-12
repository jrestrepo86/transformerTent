from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy as np
import pandas as pd
import torch
import copy


def toColVector(x):
    """
    Change vectors to column vectors
    """
    x = x.reshape(x.shape[0], -1)
    if x.shape[0] < x.shape[1]:
        x = x.T
    x.reshape((-1, 1))
    return x


class treet_data_set(Dataset):
    def __init__(
        self,
        target,
        source,
        prediction_len=8,
        history_len=1,
        normalize=None,
        source_history_len=None,
        last_x_zero=True,
    ):
        super(treet_data_set, self).__init__()
        self.prediction_len = prediction_len
        self.history_len = history_len
        self.last_x_zero = last_x_zero
        self.source_history_len = source_history_len
        self.data_len = target.shape[0] - prediction_len - history_len + 1
        if normalize is not None:
            target = (target - target.mean()) / target.std()
            source = (source - source.mean()) / source.std()

        self.target = target.clone().detach()
        self.source = source.clone().detach()

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        sequence_end = index + self.prediction_len + self.history_len

        # target = self.target[index:sequence_end].clone().detach()
        # source = self.source[index:sequence_end].clone().detach()
        target = self.target[index:sequence_end].clone().detach()
        source = self.source[index:sequence_end].clone().detach()

        if self.last_x_zero:
            source[-1].zero_()
        if self.source_history_len is not None:
            if self.source_history_len < self.history_len:
                source[: -self.source_history_len].zero_()
        return target, source


def get_apnea_data(target_name, source_name):
    data_path = "../datasets/apnea"
    varibles_name = ["heart", "breath", "oxygen"]

    df_list = []
    for file in [f for f in os.listdir(data_path) if "txt" in f]:
        file_name = f"{data_path}/{file}"
        df = pd.read_csv(
            file_name,
            sep=" ",
            header=None,
            names=varibles_name,
            index_col=False,
        )
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)
    target = np.array(df[target_name].values)
    source = np.array(df[source_name].values)
    return target, source
