from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


class Dataset_Apnea(Dataset):
    def __init__(
        self, flag="train", size=None, timeenc=0, batch_size=32, dim=1, process_info={}
    ):

        # info
        if size is None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        # assert self.label_len == 0, "Only support label_len=0"
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.batch_size = batch_size
        assert dim == 1, "Only support dim=1"
        self.dim = dim
        self.data_stride = not process_info["memory_cut"]
        self.x_type = process_info["x"]
        self.x_length = process_info["x_length"]
        self.last_x_zero = True  # zeroing the last x input when returning a batch
        # self.scale = True
        self.scale = False
        self.__read_data__()

    def __read_data__(self):
        assert os.path.isdir(
            "../datasets/apnea"
        ), "No Apnea data found in datasets/apnea."
        heart_rates = []
        breath_rates = []
        oxygen_rates = []
        for file in [f for f in os.listdir("../datasets/apnea") if "txt" in f]:
            df = pd.read_csv(
                os.path.join("../datasets/apnea", file),
                sep=" ",
                header=None,
                names=["heart", "breath", "oxygen"],
                index_col=False,
            )
            heart_rates.append(df["heart"].values)
            breath_rates.append(df["breath"].values)
            oxygen_rates.append(df["oxygen"].values)
        heart_rates = np.array(heart_rates)
        breath_rates = np.array(breath_rates)
        oxygen_rates = np.array(oxygen_rates)
        # data_raw = np.stack([heart_rate, breath_rate, oxygen_rate], axis=1) # no oxygen now...
        data_raw1 = np.stack([heart_rates[0], breath_rates[0]], axis=1)
        data_raw2 = np.stack([heart_rates[1], breath_rates[1]], axis=1)
        feature_dict = {"heart": 0, "breath": 1}  # , 'oxygen': 2}
        self.x_feature = feature_dict[self.x_type]
        indices = list(range(data_raw1.shape[1]))
        indices.remove(self.x_feature)
        indices.insert(0, self.x_feature)
        data1 = data_raw1[:, indices]
        data2 = data_raw2[:, indices]
        self.n_samples1 = len(data1)
        self.n_samples2 = len(data2)
        if self.scale:
            self.scaler1 = StandardScaler(with_std=True)
            self.scaler2 = StandardScaler(with_std=True)
            data1 = self.scaler1.fit_transform(data1)
            data2 = self.scaler2.fit_transform(data2)
        self.min_max = [
            min(data1[:, 1:].min(), data2[:, 1:].min()),
            max(data2[:, 1:].max(), data2[:, 1:].max()),
        ]
        # num_train = int(self.n_samples1 * 0.9 + self.n_samples2 * 0.9)
        # border1s = [0, num_train - self.label_len, 0]
        # border2s = [num_train, self.n_samples1 + self.n_samples2, self.n_samples1 + self.n_samples2]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]

        self.data_x = torch.tensor(np.concatenate([data1[:, :1], data2[:, :1]], axis=0))
        self.data_y = torch.tensor(np.concatenate([data1[:, 1:], data2[:, 1:]], axis=0))

        self.data_stamp = None


class Dataset_Transformer(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        if self.data_stride:
            s_begin = index * self.pred_len  # data stride is self.pred_len
        else:
            s_begin = index

        r_begin = s_begin
        r_end = s_begin + self.label_len + self.pred_len

        seq_x = self.data_x[r_begin:r_end]  # same input for encoder and decoder
        seq_y = self.data_y[r_begin:r_end]

        if hasattr(self, "last_x_zero"):
            seq_x[-1].zero_()  # for transfer entropy definition without x_t
        if hasattr(self, "x_length"):
            if self.x_length < self.label_len:
                seq_x[: -self.x_length].zero_()
        return seq_x, seq_y

    def __len__(self):
        if self.data_stride:
            return (
                len(self.data_x) - self.label_len
            ) // self.pred_len  # for data stride = self.pred_len
        else:
            return len(self.data_x) - self.label_len - self.pred_len + 1


class Dataset_Transformer_Apnea(Dataset_Transformer, Dataset_Apnea):
    def __init__(self, *args, **kwargs):
        Dataset_Transformer.__init__(self)
        Dataset_Apnea.__init__(self, *args, **kwargs)


def data_provider(args, flag):
    timeenc = 0 if args["embed"] != "timeF" else 1

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = args["batch_size"]
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = (
            True
            if args["model"] != "LSTM" or args["process_info"].get("memory_cut")
            else False
        )
        drop_last = True
        batch_size = args["batch_size"]

    data_set = Dataset_Transformer_Apnea(
        flag=flag,
        size=[args["seq_len"], args["label_len"], args["pred_len"]],
        timeenc=timeenc,
        batch_size=args["batch_size"],
        dim=args["y_dim"],
        process_info=args["process_info"],
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args["num_workers"],
        drop_last=drop_last,
    )

    return data_set, data_loader


if __name__ == "__main__":
    args = {
        "model": "Decoder_Model",
        "embed": "Fixed",
        "batch_size": 3,
        "seq_len": 0,
        "label_len": 1,
        "pred_len": 5,
        "process_info": {
            "type": "Apnea",
            "x": "breath",  # TE(heart->breath) < TE(breath->heart)
            "x_length": 3,  # -1 means all = label length (the last x is trimmed due to synchronization of the processes)
            "memory_cut": True,  # reset states of the model, and taking data without stride
        },
        "num_workers": 0,
        "y_dim": 1,
    }
    flag = "train"
    data_set, data_loader = data_provider(args, flag)

    for i, (batch_x, batch_y) in enumerate(data_loader):
        pass
        print(i)
