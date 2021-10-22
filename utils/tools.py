import numpy as np
import torch
import random


def xywh2xyxy(bbox):
    if isinstance(bbox, torch.Tensor):
        box = torch.zeros_like(bbox)
    else:
        box = np.zeros_like(bbox)
    box[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
    box[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
    box[:, 2] = bbox[:, 0] + bbox[:, 2] / 2
    box[:, 3] = bbox[:, 1] + bbox[:, 3] / 2

    return box


def xyxy2xywh(bbox):
    if isinstance(bbox, torch.Tensor):
        box = torch.zeros_like(bbox)
    else:
        box = np.zeros_like(bbox)
    box[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2
    box[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2
    box[:, 2] = (bbox[:, 2] - bbox[:, 0]) / 2
    box[:, 3] = (bbox[:, 3] - bbox[:, 1]) / 2

    return box


def use_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
