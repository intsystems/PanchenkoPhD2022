import argparse
import json
import os
import random
import numpy as np
import torch
from tqdm.auto import tqdm
from scripts.lstm.dataset import get_loaders
from scripts.lstm.lstm import LSTM
from sklearn.metrics import f1_score

data_config = {
    'data_type': 'seed',
    'signal_type': 'extract_feature_de',
    'padding': 'zeroed',
}

SEQ_LEN = 250
N_CLASSES = 3


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
    for inputs, targets in tqdm(train_loader, total=n_train, disable=tqdm_disable):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        loss_ = criterion(outputs, targets)
        loss_.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss += loss_.detach().cpu().item()
        y_true += targets.detach().cpu().tolist()
        y_pred += outputs.detach().cpu().argmax(1).tolist()

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
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            loss_ = criterion(outputs, targets)
            loss += loss_.detach().cpu().item()
            y_true += targets.detach().cpu().tolist()
            y_pred += outputs.detach().cpu().argmax(1).tolist()

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

    model = LSTM(seq_len=SEQ_LEN, output_dim=N_CLASSES)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train_loader, test_loader = get_loaders(**json.loads(args.data_config), batch_size=args.bs, shuffle=True)

    for epoch in range(1, args.num_epoch):
        loss, acc, f1 = train(train_loader, model, optimizer, criterion, device)
        print(f'TRAIN: epoch = {epoch}, loss = {round(loss, 3)}, accuracy = {round(acc, 3)}, f1 = {round(f1, 3)}')
        loss, acc, f1 = test(test_loader, model, criterion, device)
        print(f'TEST: loss = {round(loss, 3)}, accuracy = {round(acc, 3)}, f1 = {round(f1, 3)}')
        print('-' * 50)
        #if epoch == (args.num_epoch - 1):
        #    torch.save(model.state_dict(), 'data/models/seed_lstm_epoch_60_lr_1e-4.pth')