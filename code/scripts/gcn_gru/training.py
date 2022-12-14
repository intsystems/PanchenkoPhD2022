import argparse
import os
import random
import numpy as np
import torch
from tqdm.auto import tqdm
from scripts.gcn_gru.dataset import get_loaders
from scripts.gcn_gru.gcn_gru_stacked import RecurrentGCN

model_config = {
    'node_features': 20,
    'hidden_dim': 128,
    'gn_layers': 2,
    'ks': [3, 2],
    'dropout': 0.2,
    'output_dim': 3,
}

data_config = {
    'dataset': 'seed',
    'signal_type': 'extract_feature',
    'conn_method': 'dist',
    'padding': False,
}


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(epoch, train_loader, model, optimizer, criterion, device):
    model.train()
    y_true, y_pred = [], []
    loss = 0
    n_train = len(train_loader)
    for i, data in tqdm(enumerate(train_loader), total=n_train):
        data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        loss_ = criterion(out, data.y)
        loss_.backward()
        if (i + 1) % 16 == 0 or i == n_train:
            optimizer.step()
            optimizer.zero_grad()

        loss += loss_.detach().cpu().item()
        y_true += data.y.detach().cpu().tolist()
        y_pred += out.detach().cpu().argmax(1).tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    loss /= n_train
    acc = (y_true == y_pred).mean()

    print(f'TRAIN: epoch = {epoch}, loss = {round(loss, 3)}, accuracy = {round(acc, 3)}')
    print((y_pred == 0).mean(), (y_pred == 1).mean(), (y_pred == 2).mean())


def test(test_loader, model, criterion, device):
    model.eval()
    y_true, y_pred = [], []
    loss = 0
    n_test = len(test_loader)
    for i, data in enumerate(test_loader):
        data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        loss_ = criterion(out, data.y)
        loss += loss_.detach().cpu().item()
        y_true += data.y.detach().cpu().tolist()
        y_pred += out.detach().cpu().argmax(1).tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    loss /= n_test
    acc = (y_true == y_pred).mean()

    print(f'TEST: loss = {round(loss, 3)}, accuracy = {round(acc, 3)}')
    print((y_pred == 0).mean(), (y_pred == 1).mean(), (y_pred == 2).mean())
    print('-' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--num_epoch', type=int, default=51)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}')

    seed_torch()

    model = RecurrentGCN(**model_config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train_loader, test_loader = get_loaders(**data_config, batch_size=args.bs)

    for epoch in range(1, args.num_epoch):
        train(epoch, train_loader, model, optimizer, criterion, device)
        test(test_loader, model, criterion, device)
