# coding=utf-8
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import MarginRankingLoss, functional
from torch.nn.modules.loss import _Loss

from modules.common import Similarity, SimilarityEnum, get_similarity
from modules.mf_margin_loss import MarginLoss


def _no_weighting(x: torch.Tensor) -> float:
    return 1.


def _inverse_weighting(x: torch.Tensor) -> float:
    return 1. / x.shape[0]


class OptimizedMarginRankingLoss(_Loss):
    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = 'mean',
        weighting: Optional[str] = None,
    ):
        super().__init__(reduction=reduction)
        self.margin = margin
        if weighting is None:
            self.positive_weighting = None
        elif weighting == 'inverse':
            self.positive_weighting = _inverse_weighting
        else:
            raise KeyError(f'Unknown weighting scheme: "{weighting}".')

    def forward(
        self,
        selected_from_source_to_all_target: torch.FloatTensor,
        matching_target: torch.LongTensor,
    ) -> torch.FloatTensor:
        """

        :param selected_from_source_to_all_target: shape: (k, n)
        :param matching_target: shape: (k,)
        :return:
        """
        num_matches = matching_target.shape[0]
        positive = selected_from_source_to_all_target[torch.arange(num_matches), matching_target].view(num_matches, 1)

        # We do not filter out the positives, as there is exactly one positive, and the difference is 0
        negatives = selected_from_source_to_all_target

        # Margin-Ranking loss: shape: (k, n)
        loss: torch.FloatTensor = functional.relu(negatives - positive + self.margin)

        # Weighting
        if self.positive_weighting is not None:
            weight = torch.ones_like(loss)
            weight[torch.arange(num_matches), matching_target] *= self.positive_weighting(selected_from_source_to_all_target)
            loss = loss * weight

        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            raise KeyError(f'Unknown reduction: "{self.reduction}".')

        return loss


class MatchingLoss(nn.Module):
    # TODO: Split loss computation from final alignment computation
    #: The similarity
    similarity: Similarity

    def __init__(
        self,
        similarity: SimilarityEnum = SimilarityEnum.dot,
        loss=MarginLoss(),
    ):
        super().__init__()

        # Setup similarity function
        self.similarity = get_similarity(similarity=similarity)

        # Setup loss function
        self.loss_function = loss

    def forward(
        self,
        left_node_representations: torch.FloatTensor,
        right_node_representations: torch.FloatTensor,
        node_matching: torch.LongTensor,
    ) -> torch.FloatTensor:
        return self.compute_loss(
            left_node_representations=left_node_representations,
            right_node_representations=right_node_representations,
            node_matching=node_matching,
            keep_dist=False,
        )[0]

    def compute_loss(
        self,
        left_node_representations: torch.FloatTensor,
        right_node_representations: torch.FloatTensor,
        node_matching: torch.LongTensor,
        keep_dist: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        """Compute the loss for the given node matching.

        :param left_node_representations: shape: (n, d)
            The node representations of the left graph.
        :param right_node_representations: shape: (m, d)
            The node representations of the right graph.
        :param node_matching: shape: (2, k)
            Indices of matching nodes.
        :param keep_dist:
            Whether to keep the computed distances.

        :return:
            - The scalar loss.
            - left-to-all-right dist (if keep_dist is True)
            - right-to-all-left dist (if keep_dist is True)
        """
        # Compute the matching loss for the selected nodes in the first graph to all nodes in the second graph
        left_to_right_matching = node_matching
        left_to_right_loss, left_to_right_dist = self.selection_matching_loss(
            output_source=left_node_representations,
            output_target=right_node_representations,
            matching=left_to_right_matching,
            keep_dist=keep_dist,
        )

        # Compute the matching loss for the selected nodes in the first graph to all nodes in the second graph
        right_to_left_matching: torch.LongTensor = node_matching.flip(0)
        right_to_left_loss, right_to_left_dist = self.selection_matching_loss(
            output_source=right_node_representations,
            output_target=left_node_representations,
            matching=right_to_left_matching,
            keep_dist=keep_dist,
        )

        # Equally weighted combination.
        loss = left_to_right_loss + right_to_left_loss

        return loss, left_to_right_dist, right_to_left_dist

    def selection_matching_loss(
        self,
        output_source: torch.FloatTensor,
        output_target: torch.FloatTensor,
        matching: torch.LongTensor,
        keep_dist: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """Compute the loss from selected nodes in source graph to all nodes in other graph.

        :param output_source: shape: (n, d)
            The enriched node embeddings for the one graph.
        :param output_target: shape: (m, d)
            The enriched node embeddings for the other graph.
        :param matching: shape: (2, k)
            The source-to-target matching,
        :param keep_dist:
            Whether to keep the computed distances.

        :return:
            - The margin loss based on the distances from the selected nodes in the first graph to all nodes in the
              other graph.
            - The distances from selected from source to all target
        """
        n, d = output_source.shape
        m, _ = output_target.shape
        _, k = matching.shape
        assert output_target.shape[1] == d
        matching_source, matching_target = matching
        selected_output_source = output_source[matching_source]
        assert selected_output_source.shape == (k, d)
        selected_from_source_to_all_target = -self.similarity(selected_output_source, output_target)
        assert selected_from_source_to_all_target.shape == (k, m)
        source_to_target_loss = self.loss_function.forward(selected_from_source_to_all_target, matching_target)

        # Discard distances if not desired
        if not keep_dist:
            selected_from_source_to_all_target = None

        return source_to_target_loss, selected_from_source_to_all_target


class SampledMatchingLoss(MatchingLoss):
    def __init__(
        self,
        similarity: SimilarityEnum = SimilarityEnum.dot,
        pair_loss=MarginRankingLoss(),
        num_negatives: int = 1,
    ):
        super().__init__(
            similarity=similarity,
            loss=AllMarginRankingLoss(margin=getattr(pair_loss, 'margin', None), reduction=pair_loss.reduction),
        )

        # Bind parameter
        self.pair_loss = pair_loss
        self.num_negatives = num_negatives

    def forward(
        self,
        left_node_representations: torch.FloatTensor,
        right_node_representations: torch.FloatTensor,
        node_matching: torch.LongTensor,
    ) -> torch.FloatTensor:
        # Matching loss left to right
        left_to_right_mapping = node_matching
        loss_left_to_right = self._loss_source_to_target(
            source_representations=left_node_representations,
            target_representations=right_node_representations,
            source_to_target_mapping=left_to_right_mapping,
        )

        # Matching loss right to left
        right_to_left_mapping: torch.LongTensor = node_matching.flip(0)
        loss_right_to_left = self._loss_source_to_target(
            source_representations=right_node_representations,
            target_representations=left_node_representations,
            source_to_target_mapping=right_to_left_mapping,
        )

        return 0.5 * (loss_left_to_right + loss_right_to_left)

    def _loss_source_to_target(
        self,
        source_representations: torch.FloatTensor,
        target_representations: torch.FloatTensor,
        source_to_target_mapping: torch.LongTensor,
    ) -> torch.FloatTensor:
        # check representation shapes
        n_source, d1 = source_representations.shape
        n_target, d2 = target_representations.shape
        assert d1 == d2

        # Bind device
        device = source_to_target_mapping.device

        # Split mapping
        p, k = source_to_target_mapping.shape
        assert p == 2
        sources_indices, target_positive_indices = source_to_target_mapping

        # Extract representations
        source_vectors = source_representations[sources_indices]

        # Positive scores
        positive_target_vectors = target_representations[target_positive_indices]
        positive_scores = self.similarity.one_to_one(left=source_vectors, right=positive_target_vectors)
        assert positive_scores.shape == (k,)

        # Negative samples in target graph
        target_negatives_indices = torch.randint(n_target - 1, size=(k, self.num_negatives), dtype=torch.long, device=device)
        offset = (target_negatives_indices == target_positive_indices.unsqueeze(1)).long()
        target_negatives_indices = target_negatives_indices + offset

        # Negative scores
        negative_target_vectors = target_representations[target_negatives_indices]
        negative_scores = self.similarity.one_to_one(left=source_vectors.unsqueeze(1), right=negative_target_vectors)
        assert negative_scores.shape == (k, self.num_negatives)

        # Evaluate pair loss
        positive_scores = positive_scores.unsqueeze(dim=1).repeat(1, self.num_negatives).view(-1)
        negative_scores = negative_scores.view(-1)
        loss = self.pair_loss(positive_scores, negative_scores, torch.ones(k * self.num_negatives, device=device))

        return loss


class AllMarginRankingLoss(_Loss):
    def __init__(
        self,
        margin: float = 1.,
        reduction: str = 'mean',
    ):
        super().__init__(reduction=reduction)
        self.margin = margin

    def forward(
        self,
        selected_from_source_to_all_target: torch.FloatTensor,
        matching_target: torch.LongTensor,
    ) -> torch.FloatTensor:
        num_matches = matching_target.shape[0]

        # selected_from_source_to_all_target are distances
        #: shape: (k, m)
        scores = -selected_from_source_to_all_target

        #: shape: (k,)
        positive_scores = scores[torch.arange(num_matches), matching_target]

        loss = functional.relu(scores - positive_scores.unsqueeze(1) + self.margin)

        return torch.mean(loss)
