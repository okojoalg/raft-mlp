import numpy as np
import torch


class Mixup(object):
    def __init__(self, alpha=0.5):
        self.lamb = np.random.beta(alpha, alpha)

    @torch.no_grad()
    def mix(self, x, y):
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        x = self.lamb * x + (1 - self.lamb) * x[index, :]
        y1, y2 = y, y[index]
        return x, y1, y2

    def criterion(self, criterion, y_pred, y1, y2):
        return self.lamb * criterion(y_pred, y1) + (1 - self.lamb) * criterion(y_pred, y2)
