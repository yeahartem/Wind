import os
import random
import pickle
from click import pass_context
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import torch


class WindDataModule(pl.LightningDataModule):
    def __init__(
        self, X: dict, y: dict, batch_size: int = 128, downsample: bool = True
    ):
        super().__init__()
        self.batch_size = batch_size
        self.X_train, self.X_val, self.X_test = (
            torch.tensor(X["Train"], dtype=torch.double),
            torch.tensor(X["Val"], dtype=torch.double),
            torch.tensor(X["Test"], dtype=torch.double),
        )
        self.y_train, self.y_val, self.y_test = (
            torch.tensor(y["Train"], dtype=torch.double),
            torch.tensor(y["Val"], dtype=torch.double),
            torch.tensor(y["Test"], dtype=torch.double),
        )
        mean_channels = self.X_train.mean(dim=[0, -1, -2])
        std_channels = self.X_train.std(dim=[0, -1, -2])
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=mean_channels, std=std_channels),
            ]
        )

        self.dl_dict = {"batch_size": self.batch_size}

        if downsample:
            class_sample_count = [
                len(self.y_train) - sum(self.y_train),
                sum(self.y_train),
            ]
            weights = 1 / torch.Tensor(class_sample_count)
            self.sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights, self.batch_size
            )
        else:
            self.sampler = None

    def prepare_data(self):
        if type(self.y_train) == torch.Tensor and len(self.y_train.shape) == 2:
            pass
        else:
            self.y_train = torch.tensor(
                [[0.0, 1.0] if v else [1, 0.0] for v in self.y_train],
                dtype=torch.float64,
            )
            self.y_val = torch.tensor(
                [[0.0, 1.0] if v else [1.0, 0.0] for v in self.y_val],
                dtype=torch.float64,
            )
            self.y_test = torch.tensor(
                [[0.0, 1.0] if v else [1.0, 0.0] for v in self.y_test],
                dtype=torch.float64,
            )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = TensorDataset(
                self.transform(self.X_train), torch.tensor(self.y_train)
            )
            self.dataset_val = TensorDataset(
                self.transform(self.X_val), torch.tensor(self.y_val)
            )

        if stage == "test" or stage is None:
            self.dataset_test = TensorDataset(
                self.transform(self.X_test), torch.tensor(self.y_test)
            )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, sampler=self.sampler, **self.dl_dict)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, sampler=self.sampler, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, sampler=self.sampler, **self.dl_dict)


def train_val_test_split(
    path_to_data: str,
    train: float = 0.5,
    val: float = 0.25,
    test: float = 0.25,
    verbose: bool = False,
) -> dict:
    """randomly splits weather stations to train, val, test in proportions given

    Args:
        path_to_data (str): path to folder which contains folder with preparsed numpy objects from stations
        train (float, optional): train weather stations share. Defaults to 0.5.
        val (float, optional): val weather stations share. Defaults to 0.25.
        test (float, optional): test weather stations share. Defaults to 0.25.
        verbose (bool, optional): if to print out the result of split. Defaults to False.

    Returns:
        dict: [description]
    """
    stations = os.listdir(path_to_data)
    random.shuffle(stations)
    partition = {"train_share": train, "val_share": val, "test_share": test}
    train_len = int(len(stations) * partition["train_share"])
    val_len = int(len(stations) * partition["val_share"])
    test_len = int(len(stations) * partition["test_share"])
    train_sts, val_sts, test_sts = (
        stations[:train_len],
        stations[train_len : train_len + val_len],
        stations[train_len + val_len :],
    )
    st_split_dict = {"Train": train_sts, "Val": val_sts, "Test": test_sts}
    if verbose:
        print(st_split_dict)
    return st_split_dict


def extract_splitted_data(path_to_dump: str, st_split_dict: dict) -> tuple:
    """extracts X, y, splitted into train, val, test

    Args:
        path_to_dump (str): path to folder which contains folder with preparsed numpy objects from stations
        st_split_dict (dict): division by stations' names into train, val, test

    Returns:
        tuple: (X - keys = train, val, test. values = objects; y - similarly)
    """
    X = {}
    y = {}
    for split_part, sts in st_split_dict.items():
        X_split = []
        y_split = []
        for st in sts:
            st_dir = os.path.join(path_to_dump, st)
            with open(os.path.join(st_dir, "objects.pkl"), "rb") as f:
                X_ = pickle.load(f)
            X_split.append(X_)
            try:
                with open(os.path.join(st_dir, "target.pkl"), "rb") as f:
                    y_ = pickle.load(f)
                y_split.append(y_)
            except FileNotFoundError:
                y_split.append([])

        X[split_part] = np.concatenate(X_split)
        y[split_part] = np.concatenate(y_split)
    return X, y
