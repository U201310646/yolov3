import torchvision.transforms as transforms
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
from utils.tools import xywh2xyxy
import numpy as np
import torch


# Transform
# label file ： relative label(x1,x2,x3,x4)
# transform to absolute label
# padding img to height=width
# resize to a certain size (no need if transform to relative form)
# transform to relative label form
# ToTensor() for pytorch

class AbsoluteLabel:
    def __init__(self):
        pass

    def __call__(self, data):
        img, targets = data
        h, w, _ = img.shape
        # target [N,5]
        # relative x,y,w,h --> absolute x,y,w,h
        targets[:, [1, 3]] *= w
        targets[:, [2, 4]] *= h
        return img, targets


class ImgAug:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, data):
        img, targets = data
        # xywh --> x1y1x2y2
        targets[:, 1:] = xywh2xyxy(targets[:, 1:])
        bbs = BoundingBoxesOnImage([BoundingBox(*box[1:], label=box[0]) for box in targets], shape=img.shape)
        img_aug, targets = self.augmentations(image=img, bounding_boxes=bbs)
        targets = targets.clip_out_of_image()
        # targets: BoundingBox object --> numpy.array
        target_aug = np.zeros((len(targets), 5))
        for index, target in enumerate(targets):
            x1 = target.x1
            y1 = target.y1
            x2 = target.x2
            y2 = target.y2
            label = target.label

            target_aug[index, 0] = label
            target_aug[index, 1] = (x1 + x2) / 2
            target_aug[index, 2] = (y1 + y2) / 2
            target_aug[index, 3] = (x2 - x1) / 2
            target_aug[index, 4] = (y2 - y1) / 2

        return img_aug, target_aug


class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-10, 10)),
            iaa.Fliplr(0.5),
        ])


class PadSquare(ImgAug):
    # transform the images and targets
    # use imgaug API
    def __init__(self):
        self.augmentations = iaa.Sequential([iaa.PadToAspectRatio(1.0, position="center-center")]).to_deterministic()


class RelativeLabel:
    def __init__(self):
        pass

    def __call__(self, data):
        img, targets = data
        h, w, _ = img.shape
        targets[:, [1, 3]] /= w
        targets[:, [2, 4]] /= h
        return img, targets


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        img, targets = data
        img = transforms.ToTensor()(img)
        # preparation for batch info
        targets_batch = np.zeros((len(targets), 6))
        targets_batch[:, 1:] = targets
        targets_batch = torch.FloatTensor(targets_batch)

        return img, targets_batch


DEFAULT_TRANSFORM = transforms.Compose(
    [AbsoluteLabel(),
     DefaultAug(),
     PadSquare(),
     RelativeLabel(),
     ToTensor()])
