import argparse
import json
from scripts.gcn_lstm_temporal.dataset import get_loaders
from tqdm.auto import tqdm
import pickle

data_config = {
    'data_type': 'seed',
    'signal_type': 'extract_feature_de',
    'conn_method': 'corr0_overlap',
    'padding': 'zeroed',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t_cut', type=int, default=250)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--data_config', type=str, default=json.dumps(data_config))
    args = parser.parse_args()
    arg_data_config = json.loads(args.data_config)
    temporal_edge_index_temp_train, temporal_edge_weight_temp_train, temporal_batch_temp_train = [], [], []
    temporal_edge_index_temp_test, temporal_edge_weight_temp_test, temporal_batch_temp_test = [], [], []
    for t in tqdm(range(args.t_cut)):
        train_loader, test_loader = get_loaders(**arg_data_config, batch_size=args.bs, shuffle=True, t=t)
        edge_index, edge_weight, batch = [], [], []
        for x in train_loader:
            edge_index.append(x.edge_index)
            edge_weight.append(x.edge_attr)
            batch.append(x.batch)
        temporal_edge_index_temp_train.append(edge_index)
        temporal_edge_weight_temp_train.append(edge_weight)
        temporal_batch_temp_train.append(batch)

        edge_index, edge_weight, batch = [], [], []
        for x in test_loader:
            edge_index.append(x.edge_index)
            edge_weight.append(x.edge_attr)
            batch.append(x.batch)
        temporal_edge_index_temp_test.append(edge_index)
        temporal_edge_weight_temp_test.append(edge_weight)
        temporal_batch_temp_test.append(batch)

    temporal_edge_index_train, temporal_edge_weight_train, temporal_batch_train = [], [], []
    temporal_edge_index_test, temporal_edge_weight_test, temporal_batch_test = [], [], []

    num_batches = len(temporal_edge_index_temp_train[0])
    for b in range(num_batches):
        temporal_edge_index_train.append([
            t_edge_index[b] for t_edge_index in temporal_edge_index_temp_train
        ])

        temporal_edge_weight_train.append([
            t_edge_weight[b] for t_edge_weight in temporal_edge_weight_temp_train
        ])

        temporal_batch_train.append([
            t_batch[b] for t_batch in temporal_batch_temp_train
        ])

    train_graph_params = zip(temporal_edge_index_train, temporal_edge_weight_train, temporal_batch_train)

    num_batches = len(temporal_edge_index_temp_test[0])
    for b in range(num_batches):
        temporal_edge_index_test.append([
            t_edge_index[b] for t_edge_index in temporal_edge_index_temp_test
        ])

        temporal_edge_weight_test.append([
            t_edge_weight[b] for t_edge_weight in temporal_edge_weight_temp_test
        ])

        temporal_batch_test.append([
            t_batch[b] for t_batch in temporal_batch_temp_test
        ])

    test_graph_params = zip(temporal_edge_index_test, temporal_edge_weight_test, temporal_batch_test)

    with open(f'/app/data/{arg_data_config["data_type"]}/{arg_data_config["conn_method"]}_temporal_graph_params_train.pkl', 'wb') as f:
        pickle.dump(train_graph_params, f)

    with open(f'/app/data/{arg_data_config["data_type"]}/{arg_data_config["conn_method"]}_temporal_graph_params_test.pkl', 'wb') as f:
        pickle.dump(test_graph_params, f)

