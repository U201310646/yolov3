# outputs [[B,3,13,13,85][B,3,26,26,85][B,3,52,52,85]]
# targets [bath_id, class, x, y, w, h]

# build_target: relationship between anchors and gts
# the anchors need to meet the requirement(positive samples):
# the correct ratio to GT

# compute_loss: 3 parts class, bbox(for positive only), obj losses(for all samples)
# loss_cls and loss_obj: BCE_loss loss_bbox: 1-iou
import torch
import math
from utils.tools import xywh2xyxy
from torch.nn import BCEWithLogitsLoss


def box_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # box1, box2 [N, 4]
    # return [N,]
    # one to one correspondence
    if not x1y1x2y2:
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    box1_x1 = box1[:, 0]
    box1_y1 = box1[:, 1]
    box1_x2 = box1[:, 2]
    box1_y2 = box1[:, 3]
    box2_x1 = box2[:, 0]
    box2_y1 = box2[:, 1]
    box2_x2 = box2[:, 2]
    box2_y2 = box2[:, 3]

    inter = (box1_x2.clamp(max=box2_x2) - box1_x1.clamp(min=box2_x1)).clamp(min=0) * \
            (box1_y2.clamp(max=box2_y2) - box1_y1.clamp(min=box2_y1)).clamp(min=0)

    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = area2 + area1 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = (box1_x2.clamp(min=box2_x2) - box1_x1.clamp(max=box2_x1))
        ch = (box1_y2.clamp(min=box2_y2) - box1_y1.clamp(max=box2_y1))
        if DIoU or CIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            p = ((box1_x1 + box1_x2 - box2_x1 - box2_x2) ** 2 + (box1_y1 + box1_y2 - box2_y1 - box2_y2) ** 2) / 4
            if DIoU:
                return iou - p / c2

            elif CIoU:
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan((box2_x2 - box2_x1 + eps) / (box2_y2 - box2_y1 + eps)) -
                              torch.atan((box1_x2 - box1_x1 + eps) / (box2_y2 - box2_y1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (p / c2 + v * alpha)
        else:
            c = cw * ch + eps
            iou = iou - (c - union) / c
            return iou
    return iou


def compute_loss(outputs, targets, model):
    l_box = torch.zeros(1, device=targets.device)
    l_cls = torch.zeros(1, device=targets.device)
    l_obj = torch.zeros(1, device=targets.device)

    BCE_cls = BCEWithLogitsLoss()
    BCE_obj = BCEWithLogitsLoss()

    for lay_ind, layer_out in enumerate(outputs):
        t_box, indices, anchors, t_cls = build_target(targets, model)
        anchor_id, batch_id, g_i, g_j = indices[lay_ind]
        pred = layer_out[batch_id, anchor_id, g_j, g_i]

        t_obj = torch.zeros_like(layer_out[..., 4])

        if len(batch_id):
            pred_b = pred[:, :4]
            pred_b[:, :2] = torch.sigmoid(pred_b[:, :2])
            pred_b[:, 2:4] = torch.exp(pred_b[:, 2:4]) * anchors[lay_ind]

            iou = box_iou(pred_b, t_box[lay_ind], x1y1x2y2=False, CIoU=True)

            l_box += (1.0 - iou).mean()

            t_obj[batch_id, anchor_id, g_j, g_i] = iou

            pred_cls = pred[:, 5:]
            cls = torch.zeros_like(pred_cls)
            cls[range(len(t_cls[lay_ind])), t_cls[lay_ind]] = 1
            l_cls += BCE_cls(pred_cls, cls)

        l_obj += BCE_obj(layer_out[..., 4], t_obj)

    l_box *= 0.05
    l_obj *= 1.0
    l_cls *= 0.5

    loss = l_box + l_obj + l_cls

    return loss, torch.cat((l_box, l_obj, l_cls, loss))


def build_target(targets, model):
    targets_select, indices, anchor, class_select = [], [], [], []
    for index, yolo_layer in enumerate(model.yolo_layer):
        anchors = yolo_layer.anchors

        anchor_wh, anchor_id, batch_id, t_box, g_i, g_j, t_class = [], [], [], [], [], [], []

        for a_i, a_wh in enumerate(anchors):
            a_wh = a_wh / yolo_layer.stride
            ratio = targets[:, 4:6] * yolo_layer.wh / a_wh
            mask = torch.max(ratio, 1 / ratio).max(1)[0] < 4
            t = targets[mask]

            gxy = t[:, 2:4] * yolo_layer.wh
            gij = gxy.long()
            gi, gj = gij.T
            gwh = t[:, 4:6] * yolo_layer.wh

            t_box.append(torch.cat(((gxy - gij), gwh), 1))
            batch_id.append(t[:, 0])

            anchor_wh.append(a_wh.repeat(len(t), 1))
            anchor_id.append(torch.tensor(a_i).float().repeat(len(t)))

            g_i.append(gi.clamp(min=0, max=yolo_layer.wh))
            g_j.append(gj.clamp(min=0, max=yolo_layer.wh))

            c = t[:, 1].long()
            t_class.append(c)

        targets_select.append(torch.cat(t_box, 0))
        indices.append((torch.cat(anchor_id).long(), torch.cat(batch_id).long(),
                        torch.cat(g_i).long(), torch.cat(g_j).long()))
        anchor.append(torch.cat(anchor_wh, 0))
        class_select.append(torch.cat(t_class).long())

    return targets_select, indices, anchor, class_select
