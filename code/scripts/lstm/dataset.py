import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.utils import shuffle
from sklearn.model_selection import KFold


def get_data_paths(dataset, signal_type, padding='zeroed'):
    assert signal_type in (
        'raw_preprocessed',
        'extract_feature',
        'extract_feature_de',
    ), "Value of signal_type must be 'raw_preprocessed' or 'extract_feature'."

    if padding == 'last':
        x_path = f'data/{dataset}/{signal_type}/X_with_padding_last.npy'
        y_path = f'data/{dataset}/{signal_type}/y_last.npy'
    elif padding == 'zeroed':
        x_path = f'data/{dataset}/{signal_type}/X_with_padding_zeroed.npy'
        y_path = f'data/{dataset}/{signal_type}/y_zeroed.npy'
    else:
        x_path = f'data/{dataset}/{signal_type}/X_with_padding.npy'
        y_path = f'data/{dataset}/{signal_type}/y.npy'

    return x_path, y_path


class SeqDataset(Dataset):
    def __init__(self, x_path, y_path, shuff):
        super().__init__()

        self.x = np.load(x_path)
        self.y = np.load(y_path)
        self.x = self.x[:, :, 0:250, :]
        self.x = self.x.swapaxes(1, 2)
        self.x = self.x.reshape(self.x.shape[0], self.x.shape[1], -1)

        y_range = np.sort(np.unique(self.y))
        self.y = np.array([np.where(i == y_range)[0][0] for i in self.y])

        if shuff:
            self.x, self.y = shuffle(self.x, self.y, random_state=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_i = torch.from_numpy(self.x)[idx].float().T
        y_i = torch.from_numpy(self.y)[idx].long()
        return x_i, y_i


def get_loaders(data_type, signal_type, padding, batch_size, train_ratio=0.85, shuffle=False):
    data_paths = get_data_paths(data_type, signal_type, padding)
    dataset = SeqDataset(*data_paths, shuffle)

    indices = np.arange(len(dataset))
    train_size = int(len(dataset) * train_ratio)
    train_ind, test_ind = indices[:train_size], indices[train_size:]
    train_dataset, test_dataset = Subset(dataset, train_ind), Subset(dataset, test_ind)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=30)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=30)
    return train_loader, test_loader


def cross_val_loaders(
    data_type,
    signal_type,
    padding,
    batch_size,
    shuffle=False,
    k_fold=5
):
    data_paths = get_data_paths(data_type, signal_type, padding)
    dataset = SeqDataset(*data_paths, shuffle)
    indices = np.arange(len(dataset))

    loaders = []
    kf = KFold(n_splits=k_fold)
    for train_index, test_index in kf.split(indices):
        train_dataset = Subset(dataset, train_index)
        test_dataset = Subset(dataset, test_index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=30)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=30)
        loaders.append((train_loader, test_loader))
    return loaders