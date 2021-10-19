def xywh2xyxy(bbox):
    bbox[:, 0] -= bbox[:, 2] / 2
    bbox[:, 1] -= bbox[:, 3] / 2
    bbox[:, 2] += bbox[:, 0] / 2
    bbox[:, 3] += bbox[:, 1] / 2
    return bbox


def xyxy2xywh(bbox):
    bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2
    bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2
    bbox[:, 2] = (bbox[:, 3] - bbox[:, 0]) / 2
    bbox[:, 3] = (bbox[:, 3] - bbox[:, 1]) / 2
    return bbox
