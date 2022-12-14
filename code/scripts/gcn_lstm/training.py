import argparse
import os
import json
import random
import numpy as np
import torch
from tqdm.auto import tqdm
from scripts.gcn_lstm.dataset import get_loaders
from scripts.gcn_lstm.gconv_lstm import GCN_LSTM
from sklearn.metrics import f1_score

SEQ_LEN = 250
N_CLASSES = 3
NF = 5

data_config = {
    'data_type': 'seed',
    'signal_type': 'extract_feature_de',
    'conn_method': 'plv0',
    'padding': 'zeroed',
}


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(train_loader, model, optimizer, criterion, device, tqdm_disable=False):
    model.train()
    y_true, y_pred = [], []
    loss = 0
    n_train = len(train_loader)
    for data in tqdm(train_loader, total=n_train, disable=tqdm_disable):
        data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        loss_ = criterion(out, data.y)
        loss_.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss += loss_.detach().cpu().item()
        y_true += data.y.detach().cpu().tolist()
        y_pred += out.detach().cpu().argmax(1).tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    loss /= n_train
    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred, average='macro')
    return loss, acc, f1


def test(test_loader, model, criterion, device):
    model.eval()
    y_true, y_pred = [], []
    loss = 0
    n_test = len(test_loader)
    with torch.no_grad():
        for data in test_loader:
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
    f1 = f1_score(y_true, y_pred, average='macro')
    return loss, acc, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=61)
    parser.add_argument('--data_config', type=str, default=json.dumps(data_config))
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}')

    seed_torch()

    model = GCN_LSTM(node_features=NF, seq_len=SEQ_LEN, output_dim=N_CLASSES)
    model.to(device)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if 'conv1.lins' in name:
            decay.append(param)
        else:
            no_decay.append(param)

    #optimizer = torch.optim.Adam([
    #    {'params': no_decay, 'weight_decay': 0}, {'params': decay, 'weight_decay': 5e-4}
    #], lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train_loader, test_loader = get_loaders(**json.loads(args.data_config), batch_size=args.bs, shuffle=True)

    for epoch in range(1, args.num_epoch):
        loss, acc, f1 = train(train_loader, model, optimizer, criterion, device)
        print(f'TRAIN: epoch = {epoch}, loss = {round(loss, 3)}, accuracy = {round(acc, 3)}, f1 = {round(f1, 3)}')
        loss, acc, f1 = test(test_loader, model, criterion, device)
        print(f'TEST: loss = {round(loss, 3)}, accuracy = {round(acc, 3)}, f1 = {round(f1, 3)}')
        print('-' * 50)
        if epoch == (args.num_epoch - 1):
            conn = json.loads(args.data_config)['conn_method']
            torch.save(model.state_dict(), f"data/models/seed_{conn}_gcn_lstm_epoch_60_lr_1e-4.pth")