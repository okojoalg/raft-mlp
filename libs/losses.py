from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LabelSmoothingCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        alpha=0.1,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        assert 0.0 <= alpha < 1.0
        assert reduction in ["mean", "sum"]
        super(LabelSmoothingCrossEntropyLoss, self).__init__(
            weight, size_average, ignore_index, None, reduction
        )
        self.alpha = alpha

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        n = input.size()[-1]
        log_pred = F.log_softmax(input, dim=-1)
        losses = -log_pred.sum(dim=-1)
        loss = losses.mean() if self.reduction == "mean" else losses.sum()
        nll = F.nll_loss(
            log_pred,
            target,
            self.weight,
            None,
            self.ignore_index,
            None,
            self.reduction,
        )
        return self.alpha * (loss / n) + (1 - self.alpha) * nll
