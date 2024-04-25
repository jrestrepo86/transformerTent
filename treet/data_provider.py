from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import pandas as pd
import torch


def toColVector(x):
    """
    Change vectors to column vectors
    """
    x = x.reshape(x.shape[0], -1)
    if x.shape[0] < x.shape[1]:
        x = x.T
    x.reshape((-1, 1))
    return x


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

        # data to column vectors
        target = toColVector(target)
        source = toColVector(source)

        self.data_len = target.shape[0] - prediction_len - history_len + 1

        if normalize is not None:
            target = (target - target.mean()) / target.std()
            source = (source - source.mean()) / source.std()

        self.target = torch.Tensor(target).clone().detach().float()
        self.source = torch.Tensor(source).clone().detach().float()

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        sequence_end = index + self.prediction_len + self.history_len

        target = self.target[index:sequence_end].clone().detach()
        source = self.source[index:sequence_end].clone().detach()

        if self.last_x_zero:
            source[-1].zero_()  # for transfer entropy definition without x_t
        if self.source_history_len is not None:
            if self.source_history_len < self.history_len:
                source[: -self.source_history_len].zero_()
        return target, source


def data_provider(
    target,
    source,
    prediction_len,
    history_len,
    normalize_dataset,
    source_history_len,
    last_x_zero,
    batch_size,
    val_size,
    shuffle,
):

    # data to column vectors
    target = toColVector(target)
    source = toColVector(source)

    # split into data sets
    n = target.shape[0]
    val_size = int(n * val_size)
    val_inds = np.zeros_like(target, dtype=bool)
    ind_val_start = np.random.randint(0, n - val_size)
    val_inds[ind_val_start : ind_val_start + val_size] = True
    val_target_data = target[val_inds]
    val_source_data = source[val_inds]
    train_target_data = target[np.logical_not(val_inds)]
    train_source_data = source[np.logical_not(val_inds)]

    val_dataset = treet_data_set(
        val_target_data,
        val_source_data,
        prediction_len=prediction_len,
        history_len=history_len,
        normalize=normalize_dataset,
        source_history_len=source_history_len,
        last_x_zero=last_x_zero,
    )
    train_dataset = treet_data_set(
        train_target_data,
        train_source_data,
        prediction_len=prediction_len,
        history_len=history_len,
        normalize=normalize_dataset,
        source_history_len=source_history_len,
        last_x_zero=last_x_zero,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    valid_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return {
        "train": train_data_loader,
        "val": valid_data_loader,
    }
