import argparse
import os
import json
import random
import numpy as np
import torch
from tqdm.auto import tqdm
from scripts.gcn_lstm.dataset import get_loaders
from scripts.adj_learning.model import GCN_LSTM
from sklearn.metrics import f1_score

SEQ_LEN = 250
N_CLASSES = 3
NF = 5
N_CH = 62

torch.autograd.set_detect_anomaly(True)

data_config = {
    'data_type': 'seed',
    'signal_type': 'extract_feature_de',
    'conn_method': 'dist',
    'padding': True,
}

lmbd = {
    0: 1e-5,
    1: 1e3,
    2: 1,
    3: 1,
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
    ce_loss, glr_loss, spar_loss, prop_loss1, prop_loss2 = 0, 0, 0, 0, 0
    n_train = len(train_loader)
    for data in tqdm(train_loader, total=n_train, disable=tqdm_disable):
        data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        ce_loss_ = criterion(out, data.y)
        A = model.adj
        x = data.x
        bs = x.shape[0] // N_CH
        BA = torch.block_diag(*[torch.eye(len(A))] * bs).to(x.device) - torch.block_diag(*[A] * bs)

        glr_loss_ = lmbd[0] * sum([(torch.mm(torch.mm(x[..., t].T, BA), x[..., t]) / bs).pow(2).sum() \
                         for t in range(x.shape[-1])]) / x.shape[-1]
        spar_loss_ = lmbd[1] * torch.linalg.norm(A, ord=1)
        prop_loss1_ = lmbd[2] * torch.linalg.norm(A.mm(torch.ones(len(A)).unsqueeze(1).to(x.device)).squeeze(1) - torch.ones(len(A)).to(x.device)) ** 2
        prop_loss2_ = lmbd[3] * torch.abs(torch.trace(A)) ** 2
        total_loss = ce_loss_ + glr_loss_ + spar_loss_ + prop_loss1_ + prop_loss2_
        total_loss.backward()
        #print(torch.isnan(model.adj_.grad).sum())
        #print(model.adj_.grad.pow(2).sum())
        optimizer.step()
        optimizer.zero_grad()

        #print(f'ce_loss: {ce_loss}, glr_loss_: {glr_loss_}, spar_loss_: {spar_loss_}, prop_loss1_: {prop_loss1_}, prop_loss2_: {prop_loss2_}')

        ce_loss += ce_loss_.item()
        glr_loss += glr_loss_.item()
        spar_loss += spar_loss_.item()
        prop_loss1 += prop_loss1_.item()
        prop_loss2 += prop_loss2_.item()

        y_true += data.y.detach().cpu().tolist()
        y_pred += out.detach().cpu().argmax(1).tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ce_loss /= n_train
    glr_loss /= n_train
    spar_loss /= n_train
    prop_loss1 /= n_train
    prop_loss2 /= n_train

    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred, average='macro')
    return (ce_loss, glr_loss, spar_loss, prop_loss1, prop_loss2), acc, f1


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
    model = model.to(device)

    my_list = ['adj_']
    params = [x[1] for x in model.named_parameters() if x[0] in my_list]
    base_params = [x[1] for x in model.named_parameters() if x[0] not in my_list]
    optimizer = torch.optim.Adam([
        {'params': params, 'lr': 1e-3},
        {'params': base_params, 'lr': args.lr},
    ])
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train_loader, test_loader = get_loaders(**json.loads(args.data_config), batch_size=args.bs, shuffle=True)

    for epoch in range(1, args.num_epoch):
        losses, acc, f1 = train(train_loader, model, optimizer, criterion, device)
        print(f'TRAIN: epoch = {epoch}, ce_loss = {round(losses[0], 3)}, accuracy = {round(acc, 3)}, f1 = {round(f1, 3)}')
        print(f'glr_loss = {round(losses[1], 3)}, spar_loss = {round(losses[2], 3)}, prop_loss1 = {round(losses[3], 3)}, prop_loss2 = {round(losses[4], 3)}')
        loss, acc, f1 = test(test_loader, model, criterion, device)
        print(f'TEST: ce_loss = {round(loss, 3)}, accuracy = {round(acc, 3)}, f1 = {round(f1, 3)}')
        print('-' * 50)
        if (epoch % 20) == 0:
            torch.save(model.state_dict(), f'data/models/seed_adj_learning_epoch_{epoch}_lr_1e-4.pth')