# coding=utf-8
from typing import Any, Dict, Optional

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
