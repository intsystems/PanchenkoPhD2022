import argparse
import json
import torch
import numpy as np
from scripts.gcn_lstm.gconv_lstm import GCN_LSTM
from scripts.gcn_lstm.dataset import cross_val_loaders
from scripts.gcn_lstm.training import train, test, seed_torch, data_config, SEQ_LEN, N_CLASSES, NF
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=61)
    parser.add_argument('--data_config', type=str, default=json.dumps(data_config))
    parser.add_argument('--last_val_epoch', type=int, default=5)
    parser.add_argument('--k_fold', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}')

    seed_torch()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    kfold_loaders = cross_val_loaders(
        **json.loads(args.data_config),
        batch_size=args.bs,
        shuffle=True,
        k_fold=args.k_fold
    )

    train_cross_val, test_cross_val = [], []
    for k, (train_loader, test_loader) in enumerate(kfold_loaders):
        print(f'Fold = {k}:')
        model = GCN_LSTM(node_features=NF, seq_len=SEQ_LEN, output_dim=N_CLASSES)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_result, test_result = [], []
        for epoch in tqdm(range(1, args.num_epoch)):
            loss, acc, f1 = train(train_loader, model, optimizer, criterion, device, tqdm_disable=True)
            if epoch >= args.num_epoch - args.last_val_epoch:
                train_result.append((epoch, loss, acc, f1))
            loss, acc, f1 = test(test_loader, model, criterion, device)
            if epoch >= args.num_epoch - args.last_val_epoch:
                test_result.append((loss, acc, f1))

        for train_res, test_res in zip(train_result, test_result):
            print(
                'TRAIN:',
                f'epoch = {train_res[0]},',
                f'loss = {round(train_res[1], 3)},',
                f'accuracy = {round(train_res[2], 3)},',
                f'f1 = {round(train_res[3], 3)}'
            )
            print(
                'TEST:',
                f'loss = {round(test_res[0], 3)},',
                f'accuracy = {round(test_res[1], 3)},',
                f'f1 = {round(test_res[2], 3)}'
            )
            print('-' * 50)

        _, train_losses, train_metrics, train_metrics2 = list(zip(*train_result))
        train_cross_val.append((np.mean(train_losses), np.mean(train_metrics), np.mean(train_metrics2)))
        print(
            'Average train:',
            f'loss = {round(np.mean(train_losses), 3)},',
            f'accuracy = {round(np.mean(train_metrics), 3)},',
            f'f1 = {round(np.mean(train_metrics2), 3)}'
        )

        test_losses, test_metrics, test_metrics2 = list(zip(*test_result))
        test_cross_val.append((np.mean(test_losses), np.mean(test_metrics), np.mean(test_metrics2)))
        print(
            'Average test:',
            f'loss = {round(np.mean(test_losses), 3)},',
            f'accuracy = {round(np.mean(test_metrics), 3)},',
            f'f1 = {round(np.mean(test_metrics2), 3)}'
        )
        torch.save(model.state_dict(), f'data/models/cross_val/gcn_lstm/model_k{k}.pth')

    print('-' * 50)
    print('-' * 50)
    cv_losses, cv_metrics, cv_metrics2 = list(zip(*train_cross_val))
    print(
        'CROSS VAL TRAIN:',
        f'loss = {round(np.mean(cv_losses), 3)} ± {round(np.std(cv_losses), 3)},',
        f'accuracy = {round(np.mean(cv_metrics), 3)} ± {round(np.std(cv_metrics), 3)},',
        f'f1 = {round(np.mean(cv_metrics2), 3)} ± {round(np.std(cv_metrics2), 3)}'
    )

    cv_losses, cv_metrics, cv_metrics2 = list(zip(*test_cross_val))
    print(
        'CROSS VAL TEST:',
        f'loss = {round(np.mean(cv_losses), 3)} ± {round(np.std(cv_losses), 3)},',
        f'accuracy = {round(np.mean(cv_metrics), 3)} ± {round(np.std(cv_metrics), 3)},',
        f'f1 = {round(np.mean(cv_metrics2), 3)} ± {round(np.std(cv_metrics2), 3)}'
    )