import argparse
import json
import pickle
import torch
import numpy as np
from scripts.gcn_lstm.old_model import GCN_LSTM
from scripts.gcn_lstm.dataset import drop_loaders
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
    parser.add_argument('--max_drop', type=int, default=31)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}')

    seed_torch()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    dr_loaders = drop_loaders(
        **json.loads(args.data_config),
        batch_size=args.bs,
        shuffle=True,
        drop_max=args.max_drop
    )

    print(f'Total loaders = {len(dr_loaders)}.')

    train_cross_val, test_cross_val = [], []
    for k, (train_loader, test_loader) in zip([1, 2, 4, 8, 16, 32, 38, 42], dr_loaders):
        print(f'Drop count = {k}:')
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

        _, train_losses, train_metrics, train_metrics2 = list(zip(*train_result))
        train_cross_val.append((np.mean(train_losses), np.mean(train_metrics), np.mean(train_metrics2)))
        print(
            'Average train:',
            f'loss = {round(np.mean(train_losses), 3)} ± {round(np.std(train_losses), 3)},',
            f'accuracy = {round(np.mean(train_metrics), 3)} ± {round(np.std(train_metrics), 3)},',
            f'f1 = {round(np.mean(train_metrics2), 3)} ± {round(np.std(train_metrics2), 3)}'
        )

        test_losses, test_metrics, test_metrics2 = list(zip(*test_result))
        test_cross_val.append((np.mean(test_losses), np.mean(test_metrics), np.mean(test_metrics2)))
        print(
            'Average test:',
            f'loss = {round(np.mean(test_losses), 3)} ± {round(np.std(train_losses), 3)},',
            f'accuracy = {round(np.mean(test_metrics), 3)} ± {round(np.std(train_metrics), 3)},',
            f'f1 = {round(np.mean(test_metrics2), 3)} ± {round(np.std(train_metrics2), 3)}'
        )

        print('-' * 50)
    cv_losses, cv_metrics, cv_metrics2 = list(zip(*train_cross_val))
    with open('data/seed/drop_val/train_losses.pkl', 'wb') as f:
        pickle.dump(cv_losses, f)
    with open('data/seed/drop_val/train_acc.pkl', 'wb') as f:
        pickle.dump(cv_metrics, f)

    cv_losses, cv_metrics, cv_metrics2 = list(zip(*test_cross_val))
    with open('data/seed/drop_val/test_losses.pkl', 'wb') as f:
        pickle.dump(cv_losses, f)
    with open('data/seed/drop_val/test_acc.pkl', 'wb') as f:
        pickle.dump(cv_metrics, f)
