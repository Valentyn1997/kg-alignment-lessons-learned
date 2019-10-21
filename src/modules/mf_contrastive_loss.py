import torch


class ContrastiveLoss(torch.nn.modules.Module):
    """ Distance based loss function, to pull positive samples together and negative samples apart. """

    def __init__(self,
                 positive_margin=0.01,
                 negative_margin=1,
                 reduction='mean'):

        super().__init__()
        self.reduction = reduction
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)

    def forward(self, distance_matrix, rowwise_true_labels, weight_negatives=True):
        size_mapping = distance_matrix.shape[0]
        assert size_mapping == rowwise_true_labels.shape[0]

        positive_distances = distance_matrix[torch.arange(size_mapping), rowwise_true_labels]
        positive_loss = torch.sum(torch.clamp(positive_distances - self.positive_margin, min=0))
        negative_loss = torch.sum(torch.clamp(self.negative_margin - distance_matrix, min=0)) - torch.sum(torch.clamp(self.negative_margin - positive_distances, min=0))
        if weight_negatives:
            negative_weight = 1. / (distance_matrix.shape[1] - 1)
            negative_loss = negative_loss * negative_weight

        loss = positive_loss + negative_loss

        if self.reduction == 'mean':
            loss = loss / size_mapping

        return loss
