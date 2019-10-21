import gc
import logging
import os
import sys
import pprint
from enum import Enum
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Type

import humanize
import requests
import torch
import tqdm
from torch import nn, optim


def split_mapping(
    mapping: torch.LongTensor,
    train_fraction: float,
    validation_fraction: float,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """Split mapping into train-validation-test.

    :param mapping: shape: (2, n)
        The mapping between two graphs, given as pairs of indices.
    :param train_fraction: float
        The fraction of pairs to use for training.
    :param validation_fraction:
        The fraction of pairs to use for validation.

    :return:
        - train_pairs
        - val_pairs
        - test_pairs
    """
    if mapping.ndimension() != 2 or mapping.shape[0] != 2:
        raise KeyError(f'mapping is expected to be of shape (2, n), but has shape {mapping.shape}')

    # Shuffle data
    number_mappings = mapping.shape[1]
    indices = torch.randperm(number_mappings)

    # Determine split indices
    final_train_index = int(train_fraction * number_mappings)
    final_val_index = int((train_fraction + validation_fraction) * number_mappings)

    # Split mapping
    train_positives: torch.LongTensor = mapping[:, indices[:final_train_index]]
    val_positives: torch.LongTensor = mapping[:, indices[final_train_index:final_val_index]]
    test_positives: torch.LongTensor = mapping[:, indices[final_val_index:]]

    assert train_positives.shape[0] == mapping.shape[0]
    return train_positives, val_positives, test_positives


def compute_ranks(
    scores: torch.FloatTensor,
    true_indices: torch.LongTensor,
    smaller_is_better: bool = True,
    mask: Optional[torch.LongTensor] = None,
) -> torch.FloatTensor:
    """Compute the rank of the true hit.

    :param scores: shape: (k, n)
    :param true_indices: shape: (k,)
        Values between 0 (incl.) and n (excl.)
    :param smaller_is_better:
        Whether smaller of larger values are better.
    :param mask: shape: (m, 2), optional
        Optional mask for filtered setting
    :return: shape: (k,)
    """

    # Ensure that larger is better
    if smaller_is_better:
        scores = -scores

    # Get the scores of the currently considered true entity.
    batch_size = scores.shape[0]
    true_score = (scores[torch.arange(0, batch_size), true_indices.flatten()]).view(-1, 1)

    # The best rank is the rank when assuming all options with an equal score are placed behind the currently
    # considered. Hence, the rank is the number of options with better scores, plus one, as the rank is one-based.
    best_rank = (scores > true_score).sum(dim=1) + 1

    # The worst rank is the rank when assuming all options with an equal score are placed in front of the currently
    # considered. Hence, the rank is the number of options which have at least the same score minus one (as the
    # currently considered option in included in all options). As the rank is one-based, we have to add 1, which
    # nullifies the "minus 1" from before.
    worst_rank = (scores >= true_score).sum(dim=1)

    # The average rank is the average of the best and worst rank, and hence the expected rank over all permutations of
    # the elements with the same score as the currently considered option.
    avg_rank = (best_rank + worst_rank).float() * 0.5

    # In filtered setting ranking another true entity higher than the currently considered one should not be punished.
    # Hence, an adjustment is computed, which is the number of other true entities ranked higher. This adjustment is
    # subtracted from the rank.
    if mask is not None:
        batch_indices, entity_indices = mask.t()
        true_scores = true_score[batch_indices, 0]
        other_true_scores = scores[batch_indices, entity_indices]
        other_true_in_front = -(other_true_scores > true_scores).long()
        avg_rank.index_add_(dim=0, index=batch_indices, source=other_true_in_front)

    return avg_rank


def optimized_roc_auc(
    left_scores: torch.FloatTensor,
    right_scores: torch.FloatTensor,
    mapping: torch.LongTensor,
) -> float:
    """Compute ROC-AUC.

    :param left_scores: shape: (k, m)
    :param right_scores: shape: (k, n)
    :param mapping: shape: (2, k)
    """
    k, m = left_scores.shape
    _, n = right_scores.shape
    assert right_scores.shape[0] == k
    assert mapping.shape == (2, k)

    scores = torch.cat([left_scores, right_scores], dim=1).view(-1)
    total = k * (m + n)
    assert scores.shape == (total,)

    left_ind, right_ind = mapping
    left_ind += (m + n) * torch.arange(k)
    right_ind += (m + n) * torch.arange(k) + m
    ind = torch.cat([left_ind, right_ind], dim=0)
    assert ind.shape == (2 * k,)
    assert ((0 <= ind) & (ind < total)).all()

    # Determine indices of true hits in ranking
    s_ind, _ = torch.sort(torch.argsort(scores)[ind])

    # true positive rate
    tpr = torch.linspace(0, 1, steps=2 * k + 1)

    # false positive rate
    # - is decreasing
    fpr = torch.zeros(2 * k + 1)
    fpr[1:] = (total - s_ind - 2 * k + torch.arange(2 * k)).float() / total
    assert ((0 <= fpr) & (fpr <= 1)).all()

    # Compute AUC
    roc_auc = torch.sum((fpr[:-1] - fpr[1:]) * tpr[1:])

    return roc_auc.item()


def compute_ranks_for_mapping(
    left_to_all_right_dist: torch.FloatTensor,
    right_to_all_left_dist: torch.FloatTensor,
    mapping: torch.LongTensor,
) -> torch.LongTensor:
    """Compute the ranks.

    :param left_to_all_right_dist: shape: (k, m)
    :param right_to_all_left_dist: shape: (k, n)
    :param mapping: shape: (2, k)
    :return:
    """
    left_indices, right_indices = mapping
    num_left = right_to_all_left_dist.shape[1]
    num_right = left_to_all_right_dist.shape[1]
    assert left_to_all_right_dist.shape[0] == right_to_all_left_dist.shape[0]
    assert left_indices.max() < num_left
    assert right_indices.max() < num_right

    # compute ranks
    right_ranks = compute_ranks(
        scores=right_to_all_left_dist,
        true_indices=left_indices,
        smaller_is_better=True,
    )
    left_ranks = compute_ranks(
        scores=left_to_all_right_dist,
        true_indices=right_indices,
        smaller_is_better=True,
    )
    #: ranks: shape: (2*k,)
    ranks = torch.cat([left_ranks, right_ranks], dim=0)

    return ranks


def get_ranking_metrics_from_ranks(
    ranks: torch.LongTensor,
    k_values: Collection[int] = (1, 10, 50, 100),
) -> Dict[str, Any]:
    return_d = {
        'mr': torch.mean(ranks).item(),
        'mrr': torch.mean(torch.reciprocal(ranks)).item(),
    }
    for k in k_values:
        return_d[f'hits_at_{k}'] = torch.mean((ranks <= k).float()).item()
    return return_d


def evaluate_ranking_metrics(
    left_to_all_right_dist: torch.FloatTensor,
    right_to_all_left_dist: torch.FloatTensor,
    mapping: torch.LongTensor,
    k_values: Collection[int] = (1, 10, 50, 100),
) -> Dict[str, Any]:
    """Evaluate all metrics

    :param left_to_all_right_dist: shape: (k, m)
    :param right_to_all_left_dist: shape: (k, n)
    :param mapping: shape: (2, k)
    """
    ranks = compute_ranks_for_mapping(
        left_to_all_right_dist=left_to_all_right_dist,
        right_to_all_left_dist=right_to_all_left_dist,
        mapping=mapping,
    )
    return get_ranking_metrics_from_ranks(ranks=ranks, k_values=k_values)


def get_all_tensors() -> Iterable[torch.Tensor]:
    result = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                result.append(obj)
        except:
            pass
    return result


# prints currently alive Tensors and Variables
def print_alive_tensors():
    tensors = get_all_tensors()
    pprint.pprint(sorted([{
        'type': type(t),
        'size': t.shape,
        'byte': t.element_size() * t.numel(),
        'hbyte': humanize.naturalsize(t.element_size() * t.numel()),
    } for t in tensors], key=lambda d: d['byte']), width=120)

    print(f'Total: {humanize.naturalsize(sum(t.element_size() * t.numel() for t in tensors))}')


def get_param_info(module: nn.Module) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Get information about a module parameters."""
    params = module.parameters()
    num_params = 0
    num_trainable_params = 0
    data = []
    for p in params:
        data.append({
            'shape': tuple(p.shape),
            'dtype': p.dtype,
            'n_elements': p.numel(),
            'byte': p.element_size() * p.numel(),
            'trainable': p.requires_grad,
        })
        num_params += p.numel()
        num_trainable_params += (p.numel() * int(p.requires_grad))
    return num_params, num_trainable_params, data


def model_info(model: nn.Module):
    result = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            result.append([name, tuple(param.shape), humanize.naturalsize(param.numel(), gnu=True).replace('B', '')])
    return result


def _enum_values(enum_cls: Type[Enum]):
    return [v.value for v in enum_cls]


def _value_to_enum(enum_cls: Type[Enum], value: Any) -> Any:
    pos = [v for v in enum_cls if v.value == value]
    if len(pos) != 1:
        raise AssertionError(f'Could not resolve {value} for enum {enum_cls}. Available are {list(v for v in enum_cls)}.')
    return pos[0]


def truncated_normal_(tensor, mean=0, std=1):
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def download_file_from_google_drive(id, destination):
    # cf. https://stackoverflow.com/a/39225272
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def root_dir():
    # after refactoring, it is actually one folder behind src
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(root_dir.split("/")[:-2])
    return root_dir


def get_all_mask_indices(
    triples: torch.LongTensor,
    eval_batch_size: int,
) -> Dict[str, Tuple[torch.LongTensor, torch.LongTensor]]:
    logging.info('Preprocessing triples for fast filtering.')
    num_triples = triples.shape[0]
    result = dict()
    for how in ['head', 'tail']:
        cnt = 0
        split = [0]
        batch_indices = []
        entity_indices = []
        for i in tqdm.trange(0, num_triples, eval_batch_size, desc=f'Processing {how}'):
            batch = triples[i: i + eval_batch_size, :]
            batch_idx, entity_idx = get_mask_indices(batch=batch, triples=triples, how=how)
            cnt += batch_idx.shape[0]
            split.append(cnt)
            batch_indices.append(batch_idx.cpu())
            entity_indices.append(entity_idx.cpu())
        split = torch.tensor(split, dtype=torch.long)
        mask_indices = torch.stack([
            torch.cat(batch_indices, dim=0),
            torch.cat(entity_indices, dim=0),
        ], dim=1)
        num_entities = int(max(triples[:, 0].max(), triples[:, 2].max()))
        logging.info(f'Created sparse {num_triples} x {num_entities} matrix with {mask_indices.shape[0]} nnz.')
        result[how] = (split, mask_indices)
    return result


def get_mask_indices(
    batch: torch.LongTensor,
    triples: torch.LongTensor,
    how='head',
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    if how == 'head':
        same_col, diff_col = 2, 0
    else:
        same_col, diff_col = 0, 2

    #: shape: (b, t)
    # same relation
    mask = triples[None, :, 1] == batch[:, None, 1]
    #: same in other
    mask &= triples[None, :, same_col] == batch[:, None, same_col]
    #: different in how
    mask &= triples[None, :, diff_col] != batch[:, None, diff_col]

    #: (b_i, t_i) of non-zero elements, shape: (s, 2)
    batch_id, match_triple_id = mask.nonzero(as_tuple=True)

    #: shape: (?,)
    other_e = triples[match_triple_id, diff_col]

    return batch_id, other_e


def _get_optimizer_class_by_name(name: str):
    name = name.lower()
    if name == 'sgd':
        return optim.SGD
    elif name == 'adam':
        return optim.Adam
    else:
        raise KeyError(f'Unknown optimizer name: "{name}".')


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


