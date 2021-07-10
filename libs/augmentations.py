import math
import random
from typing import Tuple

import numpy as np
import torch


class Mix(object):
    def __init__(self, alpha=0.5, p=1.0):
        self.enable = np.random.choice([True, False], p=[p, 1 - p])
        self.lamb = np.random.beta(alpha, alpha) if self.enable else 1.0

    @torch.no_grad()
    def mix(self, x, y):
        raise NotImplementedError

    def criterion(self, criterion, y_pred, y: Tuple):
        if self.enable:
            return self.lamb * criterion(y_pred, y[0]) + (
                1 - self.lamb
            ) * criterion(y_pred, y[1])
        else:
            return criterion(y_pred, y[0])


class Mixup(Mix):
    def __init__(self, alpha=0.5, p=1.0):
        super().__init__(alpha, p)

    @torch.no_grad()
    def mix(self, x, y):
        if self.enable:
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).to(x.device, non_blocking=True)
            x = self.lamb * x + (1 - self.lamb) * x[index, :]
            y1, y2 = (
                (y, tuple(y_tensor[index] for y_tensor in y))
                if type(y) == tuple
                else (y, y[index])
            )
            return x, (y1, y2)
        else:
            return x, (y,)


class CutMix(Mix):
    def __init__(self, height, width, alpha=0.5, p=1.0):
        super().__init__(alpha, p)
        self.height = height
        self.width = width

    @torch.no_grad()
    def mix(self, x, y):
        if self.enable:
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).to(x.device, non_blocking=True)
            h1, h2, w1, w2 = self.get_bbox()
            x[:, :, h1:h2, w1:w2] = x[index, :, h1:h2, w1:w2]
            y1, y2 = (
                (y, tuple(y_tensor[index] for y_tensor in y))
                if type(y) == tuple
                else (y, y[index])
            )
            return x, (y1, y2)
        else:
            return x, (y,)

    def get_bbox(self):
        cut_ratio = np.sqrt(1.0 - self.lamb)
        cut_height = np.int(self.height * cut_ratio)
        cut_width = np.int(self.width * cut_ratio)
        h1 = np.random.randint(self.height - cut_height)
        w1 = np.random.randint(self.width - cut_width)
        h2 = h1 + cut_height
        w2 = w1 + cut_width
        return h1, h2, w1, w2


class Cutout(object):
    def __init__(self, p=0.5, min_area=0.02, max_area=1 / 3, aspect=0.3):
        super().__init__()
        self.p = p
        self.area = (min_area, max_area)
        self.log_aspect_ratio = (math.log(aspect), math.log(1 / aspect))
        self.min_aspect = aspect
        self.max_aspect = aspect

    @torch.no_grad()
    def __call__(self, img: torch.Tensor):
        if random.uniform(0, 1) < self.p:
            c, h, w = img.shape
            num_pixels = h * w
            area = random.uniform(*self.area) * num_pixels
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            bh = int(round(math.sqrt(area * aspect_ratio)))
            bw = int(round(math.sqrt(area / aspect_ratio)))
            for attempt in range(10):
                if bh < h and bw < w:
                    y = random.randint(h - bh)
                    x = random.randint(w - bw)
                    img[:, y : y + bh, x : x + bw] = torch.zeros(
                        (c, 1, 1), dtype=img.dtype, device=img.device
                    )
                    break
        return img
