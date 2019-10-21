# coding=utf-8
import enum
import logging
from abc import abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import Parameter, functional as F

# TODO check why this import does not work
# from utils import pairwise_similarity, pdist
from utils.torch_utils import l1c

__all__ = [
    'AverageAggregator',
    'calculate_inverse_in_degree_edge_weights',
    'KMeansClusterAggregator',
    'ClusterBasedNodeImportance',
    'DotProductSimilarity',
    'EdgeWeightsEnum',
    'FullPairwiseNodeImportance',
    'FactorizedImportance',
    'get_importance',
    'get_node_importance',
    'get_similarity',
    'GumbelSoftmaxPairwiseSimilarityToImportance',
    'ImportanceEnum',
    'LpSimilarity',
    'MaxPairwiseSimilarityToImportance',
    'Node2GraphImportance',
    'NodeImportance',
    'NodeImportanceEnum',
    'pairwise_similarity',
    'PairwiseSimilarityToImportance',
    'pdist',
    'Similarity',
    'SimilarityBasedImportance',
    'SimilarityEnum',
]


def _dist_to_sim(dist: torch.FloatTensor) -> torch.FloatTensor:
    return 1. / (1 + dist)


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    # from https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py
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


class SimilarityEnum(str, enum.Enum):
    """How to determine node/relation similarity."""
    #: Dot product
    dot = 'dot'

    #: L2-distance based
    l2 = 'l2'

    #: L1-distance based
    l1 = 'l1'


class ImportanceEnum(str, enum.Enum):
    """How to obtain the importance from the pairwise similarity matrix."""
    #: Maximum
    max_ = 'max'

    #: Gumbel softmax
    gumbel_softmax = 'gumbel_softmax'


class NodeImportanceEnum(str, enum.Enum):
    """How to determine node/relation similarity."""
    #: Full pairwise node importance (=FullPairwiseNodeImportance)
    full_pairwise = 'full_pairwise'

    #: Node to graph (= ClusterBasedNodeImportance)
    node_to_graph = 'node_to_graph'

    #: Factorized node importance (=FactorizedImportance)
    factorized = 'factorized'


class EdgeWeightsEnum(str, enum.Enum):
    """Which edge weights to use."""
    #: None
    none = 'none'

    #: Inverse in-degree -> sum of weights for incoming messages = 1
    inverse_in_degree = 'inverse_in_degree'


class Similarity(nn.Module):
    def forward(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute pairwise similarity scores.

        :param left: shape: (n, d)
        :param right: shape: (m, d)

        :return shape: (m, n)
        """
        return self.all_to_all(left=left, right=right)

    @abstractmethod
    def all_to_all(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute pairwise similarity scores.

        :param left: shape: (n, d)
        :param right: shape: (m, d)

        :return shape: (m, n)
        """
        raise NotImplementedError

    @abstractmethod
    def one_to_one(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute similarity scores.

        :param left: shape: (n, d)
        :param right: shape: (n, d)

        :return shape: (n,)
        """
        raise NotImplementedError


class DotProductSimilarity(Similarity):
    def all_to_all(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return left @ right.t()

    def one_to_one(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return torch.sum(left * right, dim=-1)


class LpSimilarity(Similarity):
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def all_to_all(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return torch.cdist(left, right, p=self.p)

    def one_to_one(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return -torch.norm(left - right, dim=-1, p=self.p)


class L1Similarity(Similarity):
    def all_to_all(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return -l1c(left, right)

    def one_to_one(
        self,
        left: torch.FloatTensor,
        right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return -torch.norm(left - right, dim=-1, p=1)


def get_similarity(
    similarity: SimilarityEnum,
) -> Similarity:
    if similarity == SimilarityEnum.dot:
        return DotProductSimilarity()
    elif similarity == SimilarityEnum.l2:
        return LpSimilarity(p=2)
    elif similarity == SimilarityEnum.l1:
        return L1Similarity()
    else:
        raise KeyError(f'Unknown similarity: {similarity}')


class PairwiseSimilarityToImportance(nn.Module):
    def forward(self, pairwise_scores: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Extract importances from pairwise scores.

        :param pairwise_scores: shape: (m, n)

        :return:
            row_importance: shape: (m, 1)
            col_importance: shape: (n, 1)
        """
        row_importance = self.left_importance(pairwise_scores=pairwise_scores)
        col_importance = self.right_importance(pairwise_scores=pairwise_scores)
        return row_importance, col_importance

    @abstractmethod
    def left_importance(self, pairwise_scores: torch.FloatTensor) -> torch.FloatTensor:
        """Extract the importance for the left side (i.e. the rows).

        :param pairwise_scores: shape: (m, n)

        :return: shape: (m, 1)
        """
        raise NotImplementedError

    @abstractmethod
    def right_importance(self, pairwise_scores: torch.FloatTensor) -> torch.FloatTensor:
        """Extract the importance for the right side (i.e. the columns).

        :param pairwise_scores: shape: (m, n)

        :return: shape: (n, 1)
        """
        raise NotImplementedError


class MaxPairwiseSimilarityToImportance(PairwiseSimilarityToImportance):
    def left_importance(self, pairwise_scores: torch.FloatTensor) -> torch.FloatTensor:
        row_importance, _ = pairwise_scores.max(dim=1)
        row_importance = row_importance.unsqueeze(-1)
        return row_importance

    def right_importance(self, pairwise_scores: torch.FloatTensor) -> torch.FloatTensor:
        col_importance, _ = pairwise_scores.max(dim=0)
        col_importance = col_importance.unsqueeze(-1)
        return col_importance


class GumbelSoftmaxPairwiseSimilarityToImportance(PairwiseSimilarityToImportance):
    def __init__(self, tau: float):
        super().__init__()
        self.tau = nn.Parameter(torch.as_tensor(tau, dtype=torch.float))

    def left_importance(self, pairwise_scores: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sum(
            F.gumbel_softmax(pairwise_scores, tau=self.tau, hard=True, dim=1) * pairwise_scores,
            dim=1,
            keepdim=True,
        )

    def right_importance(self, pairwise_scores: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sum(
            F.gumbel_softmax(pairwise_scores, tau=self.tau, hard=True, dim=0) * pairwise_scores,
            dim=0,
        ).unsqueeze(-1)


def get_importance(
    importance: ImportanceEnum,
    tau_init: Optional[float] = None,
) -> PairwiseSimilarityToImportance:
    if importance == ImportanceEnum.max_:
        return MaxPairwiseSimilarityToImportance()
    elif importance == ImportanceEnum.gumbel_softmax:
        return GumbelSoftmaxPairwiseSimilarityToImportance(tau=tau_init)
    else:
        raise KeyError(f'Unknown importance: {importance}')


class SimilarityBasedImportance(torch.nn.Module):
    def __init__(
        self,
        match_att_encoder: torch.nn.Module,
        reference_att_encoder: torch.nn.Module,
        sample_max_similarity: bool,
        tau_init: Optional[float] = None
    ):
        super(SimilarityBasedImportance, self).__init__()
        self.match_att_encoder = match_att_encoder
        self.reference_att_encoder = reference_att_encoder
        self.sample_max_similarity = sample_max_similarity
        if self.sample_max_similarity:
            assert tau_init is not None
            self.tau = Parameter(torch.tensor(float(tau_init)))

    def forward(
        self,
        match_nodes,
        reference_nodes,
    ):
        match_enc = self.match_att_encoder(match_nodes)
        ref_enc = self.reference_att_encoder(reference_nodes)
        match_reference_similarity = pairwise_similarity(match_enc, ref_enc)

        # Not necessary to apply log, same effect as applying softmax and then sample, what is  fine
        # TODO: If something breaks, it is most likely due to changing return shape from (n,) to (n, 1)
        if self.sample_max_similarity:
            match_nodes_importance = torch.sum(
                torch.nn.functional.gumbel_softmax(match_reference_similarity, dim=1, tau=self.tau, hard=1)
                * match_reference_similarity,
                dim=1,
                keepdim=True,
            )
            reference_nodes_importance = torch.sum(
                torch.nn.functional.gumbel_softmax(match_reference_similarity, dim=0, tau=self.tau, hard=1)
                * match_reference_similarity,
                dim=0,
            ).view(-1, 1)
        else:
            match_nodes_importance = match_reference_similarity.max(dim=1, keepdim=True)[0]
            reference_nodes_importance = match_reference_similarity.max(dim=0)[0].unsqueeze(-1)

        return match_nodes_importance, reference_nodes_importance


class NodeImportance(nn.Module):
    @abstractmethod
    def forward(
        self,
        match_nodes: torch.FloatTensor,
        ref_nodes: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Computes the importance score for each node in both graphs.

        :param match_nodes: shape: (n, d)
        :param ref_nodes: shape: (m, d)
        :return:
            match_node_importance: shape: (n, 1)
            ref_node_importance: shape: (m, 1)
        """
        raise NotImplementedError


class FullPairwiseNodeImportance(NodeImportance):
    def __init__(
        self,
        similarity: Similarity,
        importance: PairwiseSimilarityToImportance,
    ):
        super().__init__()
        self.similarity = similarity
        self.importance = importance

    def forward(
        self,
        match_nodes: torch.FloatTensor,
        ref_nodes: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self.importance(self.similarity.all_to_all(match_nodes, ref_nodes))


class Node2GraphImportance(NodeImportance):
    def __init__(
        self,
        similarity: Similarity,
        importance: PairwiseSimilarityToImportance,
        node_aggregator,
        ref_node_aggregator=None
    ):
        super().__init__()
        self.similarity = similarity
        self.importance = importance
        self.match_node_aggregator = node_aggregator
        if ref_node_aggregator is None:
            ref_node_aggregator = node_aggregator
        self.ref_node_aggregator = ref_node_aggregator

    def forward(
        self,
        match_nodes: torch.FloatTensor,
        ref_nodes: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        n1, d1 = match_nodes.shape
        n2, d2 = ref_nodes.shape
        assert d1 == d2

        # Aggregate node of match graph to get a graph representation
        match_graph = self.match_node_aggregator(match_nodes).view(-1, d1)

        # Compute the scores for the reference nodes based on their similarity to the match graph representation
        match_graph_to_ref_nodes = self.similarity.all_to_all(match_graph, ref_nodes)
        ref_node_importance = self.importance.right_importance(match_graph_to_ref_nodes)

        # Aggregate node of reference graph to get a graph representation
        ref_graph = self.ref_node_aggregator(ref_nodes).view(-1, d2)

        # Compute the scores for the match nodes based on their similarity to the reference graph representation
        match_nodes_to_ref_graph_similarity = self.similarity.all_to_all(match_nodes, ref_graph)
        match_node_importance = self.importance.left_importance(match_nodes_to_ref_graph_similarity)

        return match_node_importance, ref_node_importance


class AverageAggregator(nn.Module):
    def forward(self, nodes):
        return torch.mean(nodes, dim=0, keepdim=True)


class KMeansClusterAggregator(nn.Module):
    def __init__(
        self,
        num_clusters: int,
    ):
        super().__init__()
        self.num_clusters = num_clusters

    def forward(self, nodes: torch.FloatTensor) -> torch.FloatTensor:
        # 1-step k-means
        n, d = nodes.shape
        device = nodes.device

        # Randomly choose centers
        center_ind = torch.randint(d, size=(self.num_clusters,), device=device)
        centers = nodes[center_ind]

        # Compute assignment
        #: shape: (c, n)
        dist = torch.norm(nodes.unsqueeze(0) - centers.unsqueeze(1), p=2, dim=-1)
        assignment = torch.argmin(dist, dim=0)

        # Compute means
        centers = torch.zeros(self.num_clusters, d, device=device)
        torch.index_add(centers, dim=0, index=assignment, source=nodes)

        return centers


class ClusterBasedNodeImportance(Node2GraphImportance):
    def __init__(
        self,
        similarity: Similarity,
        importance: PairwiseSimilarityToImportance,
        embedding_dim: int,
        num_clusters: int,
        ref_num_clusters: Optional[int] = None,
    ):
        if ref_num_clusters is None:
            ref_num_clusters = num_clusters
        match_node_aggregator = KMeansClusterAggregator(num_clusters=num_clusters)
        ref_node_aggregator = KMeansClusterAggregator(num_clusters=ref_num_clusters)
        super().__init__(
            similarity=similarity,
            importance=importance,
            node_aggregator=match_node_aggregator,
            ref_node_aggregator=ref_node_aggregator,
        )


class FactorizedImportance(NodeImportance):
    def __init__(
        self,
        similarity: Similarity,
        embedding_dim: int,
        num_clusters: int,
        ref_num_clusters: Optional[int] = None,
    ):
        super().__init__()
        self.similarity = similarity
        if ref_num_clusters is None:
            ref_num_clusters = num_clusters
        self.match_cluster = nn.Parameter(torch.randn(num_clusters, embedding_dim))
        self.ref_cluster = nn.Parameter(torch.randn(ref_num_clusters, embedding_dim))

    def forward(
        self,
        match_nodes: torch.FloatTensor,
        ref_nodes: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        n1, d1 = match_nodes.shape
        n2, d2 = ref_nodes.shape
        assert d1 == d2

        """ The following code is more straightforward to read, but has larger memory footprint 
        
        
        # Compute similarity between node and cluster, shape: (n1, c1)
        sim_match_node_to_match_cluster = self.similarity.all_to_all(left=match_nodes, right=self.match_cluster)
        sim_match_cluster_to_ref_cluster = self.similarity.all_to_all(left=self.match_cluster, right=self.ref_cluster)
        sim_ref_cluster_to_ref_node = self.similarity.all_to_all(left=self.ref_cluster, right=ref_nodes)

        # Match mn -> mc -> rc -> rn
        mn2mc_val, mn2mc_ind = torch.max(sim_match_node_to_match_cluster, dim=1)
        mc2rc_val, mc2rc_ind = torch.max(sim_match_cluster_to_ref_cluster, dim=1)
        rc2rn_val, rc2rn_ind = torch.max(sim_ref_cluster_to_ref_node, dim=1)
        match_node_importance = mn2mc_val * mc2rc_val[mn2mc_ind] * rc2rn_val[rc2rn_ind[mc2rc_ind]]

        # Match rn -> rc -> mc -> mn
        rn2rc_val, rn2rc_ind = torch.max(sim_ref_cluster_to_ref_node, dim=0)
        rc2mc_val, rc2mc_ind = torch.max(sim_match_cluster_to_ref_cluster, dim=0)
        mc2mn_val, mc2mn_ind = torch.max(sim_match_node_to_match_cluster, dim=0)
        ref_node_importance = rn2rc_val * rc2mc_val[rn2rc_ind] * mc2mn_val[rc2mc_ind[rc2rn_ind]]
        
        
        """
        sim_match_node_to_match_cluster = self.similarity.all_to_all(left=match_nodes, right=self.match_cluster)
        match_node_importance, mn2mc_ind = torch.max(sim_match_node_to_match_cluster, dim=1)
        mc2mn_val, mc2mn_ind = torch.max(sim_match_node_to_match_cluster, dim=0)
        sim_match_cluster_to_ref_cluster = self.similarity.all_to_all(left=self.match_cluster, right=self.ref_cluster)
        mc2rc_val, mc2rc_ind = torch.max(sim_match_cluster_to_ref_cluster, dim=1)
        match_node_importance *= mc2rc_val[mn2mc_ind]
        rc2mc_val, rc2mc_ind = torch.max(sim_match_cluster_to_ref_cluster, dim=0)
        sim_ref_cluster_to_ref_node = self.similarity.all_to_all(left=self.ref_cluster, right=ref_nodes)
        rc2rn_val, rc2rn_ind = torch.max(sim_ref_cluster_to_ref_node, dim=1)
        match_node_importance *= rc2rn_val[mc2rc_ind[mn2mc_ind]]
        rn2rc_val, rn2rc_ind = torch.max(sim_ref_cluster_to_ref_node, dim=0)
        ref_node_importance = rn2rc_val * rc2mc_val[rn2rc_ind] * mc2mn_val[rc2mc_ind[rn2rc_ind]]

        return match_node_importance.unsqueeze(-1), ref_node_importance.unsqueeze(-1)


def get_node_importance(
    node_importance: NodeImportanceEnum,
    similarity: SimilarityEnum,
    importance: ImportanceEnum,
    tau_init: Optional[float] = None,
    embedding_dim: Optional[int] = None,
    num_clusters: Optional[int] = None,
    ref_num_clusters: Optional[int] = None,
) -> NodeImportance:
    importance = get_importance(importance=importance, tau_init=tau_init)
    similarity = get_similarity(similarity=similarity)

    if node_importance in {NodeImportanceEnum.factorized, NodeImportanceEnum.node_to_graph} and num_clusters is None:
        logging.warning('No number of clusters specified. Setting it to 32')
        num_clusters = 32

    if node_importance == NodeImportanceEnum.full_pairwise:
        return FullPairwiseNodeImportance(similarity=similarity, importance=importance)
    elif node_importance == NodeImportanceEnum.node_to_graph:
        if embedding_dim is None:
            raise KeyError(f'Cannot choose {node_importance} with embedding_dim=None.')
        return ClusterBasedNodeImportance(similarity=similarity, importance=importance, embedding_dim=embedding_dim, num_clusters=num_clusters, ref_num_clusters=ref_num_clusters)
    elif node_importance == node_importance.factorized:
        if embedding_dim is None:
            raise KeyError(f'Cannot choose {node_importance} with embedding_dim=None.')
        return FactorizedImportance(similarity=similarity, embedding_dim=embedding_dim, num_clusters=num_clusters, ref_num_clusters=ref_num_clusters)
    else:
        raise KeyError(f'Unknown node importance: {node_importance}.')


def calculate_inverse_in_degree_edge_weights(edge_tensor: torch.LongTensor) -> torch.FloatTensor:
    target = edge_tensor[1]
    _, inverse, counts = torch.unique(target, return_counts=True, return_inverse=True)
    edge_weights = torch.reciprocal(counts.float())[inverse].unsqueeze(dim=1)
    return edge_weights
