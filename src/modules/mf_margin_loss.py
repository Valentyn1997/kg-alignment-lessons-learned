from modules.mf_contrastive_loss import ContrastiveLoss


class MarginLoss(ContrastiveLoss):
    # (12) from https://arxiv.org/abs/1904.12787
    """ Distance based loss function, to pull positive samples together and negative samples apart. """

    def __init__(self,
                 margin=0.01,
                 reduction='mean'):
        super(MarginLoss, self).__init__(
            positive_margin=1. - margin,
            negative_margin=1. + margin,
            reduction=reduction)
