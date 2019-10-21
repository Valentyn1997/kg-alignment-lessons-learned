# coding=utf-8
import json
import random
import logging
from pprint import pprint, pformat
import argparse
import itertools
import hashlib

import humanize
import numpy as np
import torch
import tqdm
import mlflow
import mlflow.pytorch
from torch import optim
from torch.nn import MarginRankingLoss, init
from utils.torch_utils import evaluate_model

from data.knowledge_graph import get_dataset_by_name
from modules.common import EdgeWeightsEnum, SimilarityEnum
from modules.gcn_align import GCNAlign
from modules.losses import SampledMatchingLoss
from utils.common import get_param_info
from utils.mlflow_utils import log_config_to_mlflow, log_metrics_to_mlflow, connect_mlflow, _to_dot


def main():
    logging.basicConfig(level=logging.INFO)

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_mlflow', dest='log_to_mlflow', default=False, action='store_true')
    parser.add_argument('--dataset_name', type=str, default='dwy100k')
    parser.add_argument('--subset_name', type=str, default='zh_en')
    parser.add_argument('--model_class', type=str, default='GCNAlign')
    parser.add_argument('--num_epochs', type=int, default=2000)  # 50, 2000, 6000
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--num_negatives', type=int, default=5)  # 200
    parser.add_argument('--lr', type=float, default=20)  # 1.0e-03, 1.0e-02
    parser.add_argument('--use_conv_weights', type=bool, default=True)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # Mlflow settings
    print(args.log_to_mlflow)
    if args.log_to_mlflow:
        connect_mlflow('http://localhost:5000')
        mlflow.set_experiment(args.model_class)

    # Set random seed
    seed = 12306
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 1. Extensive hyperparameter search for one subset of one dataser (dbp15k_jape - zh_en)
    # params_search = {
    #     'dataset_name': [args.dataset_name],
    #     'subset_name': [args.subset_name],
    #     'num_epochs': [10, 500, 2000, 3000],
    #     'lr': [0.1, 0.5, 1, 10, 20],
    #     'eval_batch_size': [1000],  # 1000
    #     'embedding_dim': [200],
    #     'n_layers': [1, 2, 3],
    #     'num_negatives': [5, 50, 100],
    #     'use_edge_weights': [EdgeWeightsEnum.inverse_in_degree],  # None
    #     'use_conv_weights': [True, False],
    #     'conv_weight_init': [init.xavier_uniform_],
    #     'train_val_ratio': [0.7, 0.8],  # size of train subset in comparison with
    #     'node_embedding_init': ['total', 'individual'],
    #     'optimizer': [optim.SGD, optim.Adam],
    #     'seed': [seed]
    # }

    # 2. Extensive hyperparameter search for all datasets and subsets
    subsets = {
        'dwy100k': ['wd', 'yg'],
        'wk3l15k': ['en_de', 'en_fr'],
        'wk3l120k': ['en_de', 'en_fr'],
        'dbp15k_full': ['zh_en', 'ja_en', 'fr_en'],
        'dbp15k_jape': ['zh_en', 'ja_en', 'fr_en'],  # zh_en' already evaluated in 1
    }
    #
    params_search = {
        'dataset_name': [args.dataset_name],
        'subset_name': subsets[args.dataset_name],
        'num_epochs': [2000],
        'lr': [1, 10, 20, 30],
        'eval_batch_size': [1000],  # 1000
        'embedding_dim': [200],
        'n_layers': [2, 3, 4],
        'num_negatives': [50],
        'use_edge_weights': [EdgeWeightsEnum.inverse_in_degree],  # None
        'use_conv_weights': [True],
        'vertical_sharing': [False],  # [True, False]
        'conv_weight_init': [init.xavier_uniform_],
        'train_val_ratio': [0.8],  # size of train subset in comparison with
        'node_embedding_init': ['total', 'none'],  # Cite normalisation constant
        'optimizer': [optim.Adam],
        'seed': [seed]
    }

    # params_search = {
    #     'dataset_name': [args.dataset_name],
    #     'subset_name': subsets[args.dataset_name],
    #     'num_epochs': [2000, 3000],
    #     'lr': [0.5, 1],
    #     'eval_batch_size': [1000],  # 1000
    #     'embedding_dim': [200],
    #     'n_layers': [2],
    #     'num_negatives': [100],
    #     'use_edge_weights': [EdgeWeightsEnum.inverse_in_degree],  # None
    #     'use_conv_weights': [False, True],
    #     'conv_weight_init': [init.xavier_uniform_],
    #     'train_val_ratio': [0.8],  # size of train subset in comparison with
    #     'node_embedding_init': ['total'],  # Cite normalisation constant
    #     'optimizer': [optim.SGD],
    #     'seed': [seed]
    # }

    # 3. Retraining best param and non-param models on all the datasets
    # ​
    # params_search = {
    #     'dataset_name': [args.dataset_name],
    #     'subset_name': subsets[args.dataset_name],
    #     'num_epochs': [2000, 3000],
    #     'lr': [0.5, 1],
    #     'eval_batch_size': [1000],  # 1000
    #     'embedding_dim': [200],
    #     'n_layers': [2],
    #     'num_negatives': [100],
    #     'use_edge_weights': [EdgeWeightsEnum.inverse_in_degree],  # None
    #     'use_conv_weights': [False, True],
    #     'conv_weight_init': [init.xavier_uniform_],
    #     'train_val_ratio': [0.8],  # size of train subset in comparison with
    #     'node_embedding_init': ['none'],  # Cite normalisation constant
    #     'optimizer': [optim.SGD],
    #     'seed': [seed]
    # }

    # params_best_nonpar = {
    #     'dataset_name': [args.dataset_name],
    #     'subset_name': subsets[args.dataset_name],
    #     'num_epochs': [2000],
    #     'lr': [1],
    #     'eval_batch_size': [1000],  # 1000
    #     'embedding_dim': [200],
    #     'n_layers': [2],
    #     'num_negatives': [50],
    #     'use_edge_weights': [EdgeWeightsEnum.inverse_in_degree],  # None
    #     'use_conv_weights': [False],
    #     'conv_weight_init': [init.xavier_uniform_],
    #     'train_val_ratio': [1.0],  # size of train subset in comparison with
    #     'node_embedding_init': ['none'],  # Cite normalisation constant
    #     'optimizer': [optim.Adam],
    #     'seed': [seed]
    # }
    #
    # params_best_par = {
    #     'dataset_name': [args.dataset_name],
    #     'subset_name': subsets[args.dataset_name],
    #     'num_epochs': [2000],
    #     'lr': [1],
    #     'eval_batch_size': [1000],  # 1000
    #     'embedding_dim': [200],
    #     'n_layers': [2],
    #     'num_negatives': [100],
    #     'use_edge_weights': [EdgeWeightsEnum.inverse_in_degree],  # None
    #     'use_conv_weights': [True],
    #     'conv_weight_init': [init.xavier_uniform_],
    #     'train_val_ratio': [1.0],  # size of train subset in comparison with
    #     'node_embedding_init': ['total'],
    #     'optimizer': [optim.SGD],
    #     'seed': [seed]
    # }


    params_search_list = [dict(zip(params_search.keys(), values)) for values in itertools.product(*params_search.values())]

    # params_list_nonpar = [dict(zip(params_best_nonpar.keys(), values)) for values in itertools.product(*params_best_nonpar.values())]
    # params_list_par = [dict(zip(params_best_par.keys(), values)) for values in itertools.product(*params_best_par.values())]
    # params_search_list = params_list_nonpar + params_list_par

    for run, params in enumerate(params_search_list):
        logging.info(f'================== Run {run}/{len(params_search_list)} ==================')

        # Check, if run with current parameters already exists
        query = ' and '.join(list(map(lambda item: f"params.{item[0]} = '{str(item[1])}'", _to_dot(params).items())))
        print(query)
        run_hash = hashlib.md5(query.encode()).hexdigest()
        params['run_hash'] = run_hash
        if args.log_to_mlflow:
            existing_runs = mlflow.search_runs(filter_string=f"params.run_hash = '{run_hash}'", run_view_type=mlflow.tracking.client.ViewType.ACTIVE_ONLY)
            if len(existing_runs) > 0:
                logging.info('Skipping existing run.')
                continue

        mlflow.start_run() if args.log_to_mlflow else None

        # Log to MLFLOW
        log_config_to_mlflow(params) if args.log_to_mlflow else None

        # Dataset name validation
        dataset = get_dataset_by_name(dataset_name=params['dataset_name'], subset_name=params['subset_name'], inverse_triples=True, split='30', self_loops=True)
        match_triples, ref_triples, ea_train, ea_test = dataset.match_triples, dataset.ref_triples, dataset.entity_alignment_train, dataset.entity_alignment_test

        # Train -> Train + Validation
        split = int(params['train_val_ratio'] * ea_train.shape[1])
        indices = list(range(ea_train.shape[1]))
        np.random.shuffle(indices)
        ea_train, ea_val = ea_train[:, indices[:split]],  ea_train[:, indices[split:]]

        match_edge_tensor, ref_edge_tensor = [torch.stack([t[:, 0], t[:, 2]], dim=0) for t in (match_triples, ref_triples)]
        num_match_nodes, num_ref_nodes = [int(t.max() + 1) for t in (match_edge_tensor, ref_edge_tensor)]
        logging.info(f'#alignments train: {ea_train.shape}. #alignments val: {ea_val.shape}')

        # Model init
        model = eval(args.model_class)(
            num_match_nodes=num_match_nodes,
            num_ref_nodes=num_ref_nodes,
            match_edge_tensor=match_edge_tensor,
            ref_edge_tensor=ref_edge_tensor,
            device=device,
            **params,
        )
        total_params, trainable_params, _ = get_param_info(model)
        # param_string = "\n".join(map(str, model_info(model)))
        params['total_params'] = total_params
        params['trainable_params'] = trainable_params
        logging.info(f'Parameters:\n {pformat(params)}')
        logging.info(f'In total {humanize.naturalsize(total_params, gnu=True)} parameters, from which {humanize.naturalsize(trainable_params, gnu=True)} are trainable.')
        # Send to device
        ea_train = ea_train.to(device=device)
        ea_val = ea_val.to(device)
        model = model.to(device=device)
        logging.info(f'Model:\n{model}')

        # Optimizer and loss
        loss = SampledMatchingLoss(num_negatives=params['num_negatives'], similarity=SimilarityEnum.l1, pair_loss=MarginRankingLoss(margin=3.))
        optimizer = params['optimizer'](
            params=model.parameters(),
            lr=params['lr'],
        )

        # Log to MLFLOW
        log_config_to_mlflow(params) if args.log_to_mlflow else None


        # Training
        epochs = tqdm.trange(params['num_epochs'], desc=f'Training on {device}', unit='epoch', unit_scale=True)
        for e in epochs:
            # Train phase
            model.train()
            node_repr = model()
            train_loss = loss(*node_repr, ea_train)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Validation phase
            model.eval()
            val_loss = loss(*node_repr, ea_val)

            # Logging
            metrics = {
                'l1_match_train': train_loss.item(),
                'l1_match_val': val_loss.item(),
            }
            log_metrics_to_mlflow(metrics, step=e) if args.log_to_mlflow else None
            epochs.set_postfix(metrics)

        similarity = loss.similarity
        alignments = dict(zip(['train', 'val', 'test'], [ea_train, ea_val, ea_test]))
        evaluation = evaluate_model(model, alignments, similarity, params['eval_batch_size'], device)
        pprint(evaluation, indent=2, width=120)

        # Logging to MLFLOW
        if args.log_to_mlflow:
            log_metrics_to_mlflow(evaluation, params['num_epochs'])
            # mlflow.pytorch.log_model(model, 'model')
            mlflow.end_run()


if __name__ == '__main__':
    main()
