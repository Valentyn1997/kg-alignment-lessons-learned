import timeit
from typing import Collection, Type

import torch
from torch import nn
from torch.nn.init import uniform_


#: analogous to matching function described in https://arxiv.org/pdf/1905.11605.pdf Appendix A

class MultiheadCosineMatching(torch.nn.Module):
    def __init__(self, input, heads):
        super(MultiheadCosineMatching, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(heads, 1, input))
        self.reset_parameters()

    def reset_parameters(self):
        uniform_(self.weight)

    def forward(self,
                match_g: torch.Tensor,
                ref_g: torch.Tensor):
        match_processed = match_g.unsqueeze(0) * self.weight
        ref_processed = ref_g.unsqueeze(0) * self.weight
        similarity_per_head = torch.nn.functional.cosine_similarity(match_processed, ref_processed, dim=2)
        return torch.t(similarity_per_head)


class Hadamard(torch.nn.Module):

    def forward(self,
                match_g,
                ref_g):
        return match_g * ref_g


# from https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py
def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def pairwise_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise similarity between any element in a and any element in b."""
    return 1. / (1. + pdist(a, b))


_ACTIVATION_NAME_TO_CLASS = {
    cls.__name__.lower(): cls for cls in (
        nn.ELU,
        nn.Hardshrink,
        nn.Hardtanh,
        nn.LeakyReLU,
        nn.LogSigmoid,
        nn.PReLU,
        nn.ReLU,
        nn.RReLU,
        nn.SELU,
        nn.CELU,
        nn.Sigmoid,
        nn.Softplus,
        nn.Softshrink,
        nn.Softsign,
        nn.Tanh,
        nn.Tanhshrink,
    )
}


def get_activation_class_by_name(activation_cls_name: str) -> Type[nn.Module]:
    key = activation_cls_name.lower()
    if key not in _ACTIVATION_NAME_TO_CLASS.keys():
        raise KeyError(f'Unknown activation class name: {key} not in {_ACTIVATION_NAME_TO_CLASS.keys()}.')
    activation_cls_name = _ACTIVATION_NAME_TO_CLASS[key]
    return activation_cls_name


def weight_reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def get_rank(sim: torch.FloatTensor, true: torch.LongTensor) -> torch.FloatTensor:
    batch_size = true.shape[0]
    true_sim = sim[torch.arange(batch_size), true].unsqueeze(1)
    best_rank = torch.sum(sim > true_sim, dim=1, dtype=torch.long).float() + 1
    worst_rank = torch.sum(sim >= true_sim, dim=1, dtype=torch.long).float()
    return 0.5 * (best_rank + worst_rank)


def evaluate_model(model, alignments, similarity, eval_batch_size, device, ks: Collection[int] = (1, 10, 50, 100)):
    start = timeit.default_timer()

    # Evaluation
    with torch.no_grad():

        # Set model in evaluation mode
        model.eval()

        a, b = model()

        return_d = dict()
        for name, ea in alignments.items():
            num_alignments = ea.shape[1]
            ea = ea.to(device=device)

            all_left, all_right = ea
            ranks = torch.empty(2, num_alignments)

            for i in range(0, num_alignments, eval_batch_size):
                left, right = ea[:, i:i + eval_batch_size]
                num_match = left.shape[0]
                true = torch.arange(i, i + num_match, dtype=torch.long, device=a.device)

                sim_right_to_all_left = similarity.all_to_all(a[all_left], b[right]).t()
                ranks[0, i:i + eval_batch_size] = get_rank(sim=sim_right_to_all_left, true=true)

                sim_left_to_all_right = similarity.all_to_all(a[left], b[all_right])
                ranks[1, i:i + eval_batch_size] = get_rank(sim=sim_left_to_all_right, true=true)

            return_d[f'{name}_mr'] = torch.mean(ranks).item()
            return_d[f'{name}_mrr'] = torch.mean(torch.reciprocal(ranks)).item()
            for k in ks:
                return_d[f'{name}_hits_at_{k}'] = torch.mean((ranks <= k).float()).item()

    end = timeit.default_timer()
    # print(f'Evaluation took {(end - start):.5f} seconds.')

    return return_d


# TODO: Workaround until https://github.com/pytorch/pytorch/issues/24345 is fixed
# Inherit from Function
class L1CDist(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        ctx.save_for_backward(x1, x2)

        # cdist.forward does not have the memory problem
        return torch.cdist(x1, x2, p=1)

    @staticmethod
    def backward(ctx, grad_dist):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        grad_x1 = grad_x2 = None

        # Retrieve saved values
        x1, x2 = ctx.saved_tensors
        dims = x1.shape[1]

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_x1 = torch.empty_like(x1)
        if ctx.needs_input_grad[1]:
            grad_x2 = torch.empty_like(x2)

        if any(ctx.needs_input_grad):
            for i in range(dims):
                #: sign: shape: (n1, n2)
                sign = torch.sign(x1[:, None, i] - x2[None, :, i])
                if ctx.needs_input_grad[0]:
                    grad_x1[:, i] = torch.sum(grad_dist * sign, dim=1)
                if ctx.needs_input_grad[1]:
                    grad_x2[:, i] = -torch.sum(grad_dist * sign, dim=0)

        return grad_x1, grad_x2


l1c = L1CDist.apply

# # TODO: Workaround until https://github.com/pytorch/pytorch/issues/24345 is fixed
# # Inherit from Function
# class MaxL1Matching(nn.Function):
#
#     @staticmethod
#     # bias is an optional argument
#     def forward(ctx, x1, x2):
#         dist = torch.cdist(x1, x2, p=1)
#         row_match_value, row_match_idx = torch.min(dist, dim=0)
#         col_match_value, col_match_idx = torch.min(dist, dim=1)
#         d = x1.shape[1]
#         ctx.save_for_backward(row_match_value, row_match_idx, col_match_value, col_match_idx, d)
#         return row_match_value, col_match_value
#
#     @staticmethod
#     def backward(ctx, grad_row_match_value, grad_col_match_value):
#         # This is a pattern that is very convenient - at the top of backward
#         # unpack saved_tensors and initialize all gradients w.r.t. inputs to
#         # None. Thanks to the fact that additional trailing Nones are
#         # ignored, the return statement is simple even when the function has
#         # optional inputs.
#         row_match_value, row_match_idx, col_match_value, col_match_idx, d = ctx.saved_tensors
#         n1, n2 = row_match_value.shape[0], col_match_value.shape[0]
#         grad_x1 = grad_x2 = None
#
#         # These needs_input_grad checks are optional and there only to
#         # improve efficiency. If you want to make your code simpler, you can
#         # skip them. Returning gradients for inputs that don't require it is
#         # not an error.
#         if ctx.needs_input_grad[0]:
#             grad_x1 = torch.zeros(n1, d)
#             # TODO: Fixme
#         if ctx.needs_input_grad[1]:
#             grad_x2 = torch.zeros(n2, d)
#             # TODO: Fixme
#
#         return grad_x1, grad_x2
