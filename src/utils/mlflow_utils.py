# coding=utf-8
from typing import Any, Dict, Optional
import pandas as pd
import mlflow

# Make sure to import this file *BEFORE* running experiments
def connect_mlflow(tracking_uri='http://stormheim.dbs.ifi.lmu.de:5000'):
    mlflow.set_tracking_uri(tracking_uri)


def _to_dot(config: Dict[str, Any], prefix=None) -> Dict[str, Any]:
    result = dict()
    for k, v in config.items():
        if prefix is not None:
            k = f'{prefix}.{k}'
        if isinstance(v, dict):
            v = _to_dot(v, prefix=k)
        elif hasattr(v, '__call__'):
            v = {k: v.__name__}
        else:
            v = {k: v}
        result.update(v)
    return result


def log_config_to_mlflow(config: Dict[str, Any]):
    nice_config = _to_dot(config)
    mlflow.log_params(nice_config)


def log_metrics_to_mlflow(metrics: Dict[str, Any], step: Optional[int] = None):
    nice_metrics = _to_dot(metrics)
    mlflow.log_metrics(nice_metrics, step=step)


def drop_duplicates_and_zero_NA(results_df):
    results_df = results_df.drop_duplicates(subset=['params.run_hash'])

    # Zeroing NAN and other non-converged results
    results_df.loc[results_df['metrics.l1_match_val'].isna() | (results_df['metrics.test_hits_at_1'] == 1.0),
                   ['metrics.l1_match_train', 'metrics.l1_match_val',
                    'metrics.train_mr', 'metrics.test_mr', 'metrics.test_hits_at_100',
                    'metrics.train_hits_at_100', 'metrics.train_hits_at_10',
                    'metrics.val_hits_at_10', 'metrics.val_hits_at_100',
                    'metrics.train_hits_at_50', 'metrics.test_hits_at_50', 'metrics.val_mr',
                    'metrics.test_mrr', 'metrics.train_mrr', 'metrics.val_hits_at_1',
                    'metrics.train_hits_at_1', 'metrics.val_mrr', 'metrics.test_hits_at_1',
                    'metrics.test_hits_at_10', 'metrics.val_hits_at_50']] = 0.0
    return results_df


def get_best_runs_params_ablation(
        split_by=('params.dataset_name', 'params.subset_name', 'params.node_embedding_init', 'params.use_conv_weights'),
        sort_by='metrics.val_hits_at_1'):
    connect_mlflow('http://localhost:5000')
    results = mlflow.search_runs(experiment_ids=[1])
    results = drop_duplicates_and_zero_NA(results)
    results1 = results[results['params.lr'].isin(['1', '10', '20', '30']) &
                       results['params.n_layers'].isin(['2', '3', '4']) &
                       (results['params.num_negatives'] == '50') &
                       (results['params.num_epochs'] == '2000') &
                       (results['params.train_val_ratio'] == '0.8') &
                       (results['params.optimizer'] == 'Adam')]

    results2 = results[results['params.lr'].isin(['0.5', '1']) &
                       results['params.n_layers'].isin(['2']) &
                       (results['params.num_negatives'] == '100') &
                       (results['params.num_epochs'].isin(['2000', '3000'])) &
                       (results['params.train_val_ratio'] == '0.8') &
                       (results['params.optimizer'] == 'SGD') &
                       results['params.node_embedding_init'].isin(['total', 'none'])]

    results_abl = pd.concat([results1, results2])
    results_abl = results_abl.sort_values(sort_by, ascending=False).drop_duplicates(split_by)
    results_abl = results_abl.filter(regex='params\\..+')
    results_abl = results_abl.rename(columns=lambda col_name: col_name.replace('params.', ''))
    results_abl = results_abl.drop(columns=['trainable_params', 'run_hash', 'vertical_sharing', 'total_params',
                                            'seed', 'train_val_ratio'])

    # Transformations
    def import_eval(val, import_str):
        exec(import_str)
        return eval(val)

    results_abl.use_conv_weights = results_abl.use_conv_weights.apply(eval)
    results_abl.conv_weight_init = results_abl.conv_weight_init.apply(lambda val: import_eval(val, 'from torch.nn.init import xavier_uniform_'))
    results_abl.optimizer = results_abl.optimizer.apply(lambda val: import_eval(val, 'from torch.optim import Adam, SGD'))
    results_abl.use_edge_weights = results_abl.use_edge_weights.apply(lambda val: import_eval(val, 'from modules.common import EdgeWeightsEnum'))
    results_abl.embedding_dim = results_abl.embedding_dim.astype(int)
    results_abl.eval_batch_size = results_abl.eval_batch_size.astype(int)
    results_abl.num_epochs = results_abl.num_epochs.astype(int)
    results_abl.n_layers = results_abl.n_layers.astype(int)
    results_abl.num_negatives = results_abl.num_negatives.astype(int)
    results_abl.lr = results_abl.lr.astype(float)
    return results_abl


