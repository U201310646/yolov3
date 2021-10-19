import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.transform import DEFAULT_TRANSFORM
import os
import cv2


def resize(img, size):
    return F.interpolate(img.unsqueeze(0), size, mode='nearest').squeeze(0)


class CustomDataset(Dataset):
    def __init__(self, path, imgsize=416, mutiscale=False, transform=None):
        self.labels_file = []
        with open(path, 'r') as f:
            self.img_file = f.readlines()
        for path in self.img_file:
            label_path = path.replace('images', 'labels')
            label_path = os.path.splitext(label_path)[0] + '.txt'
            self.labels_file.append(label_path)
        self.img_size = imgsize
        self.multiscale = mutiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.transform = transform
        self.max_objects = 100
        self.batch_count = 0

    def __getitem__(self, item):
        try:
            img_path = self.img_file[item].rstrip()
            image = np.array(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), dtype=np.uint8)

        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        try:
            label_path = self.labels_file[item].rstrip()
            target = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        if self.transform:
            try:
                img, target = self.transform((image, target))

            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, target

    def collate_fn(self, batch):
        self.batch_count += 1
        img_path, images, targets = list(zip(*batch))
        for batch_i, target in enumerate(targets):
            target[:, 0] = batch_i
        targets = torch.cat(targets, 0)

        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size, 32))

        images = torch.stack([resize(img, self.img_size) for img in images], 0)
        return img_path, images, targets

    def __len__(self):
        return len(self.img_file)


def run():
    from utils.parse_config import parse_data_config
    data_config = "config/coco.data"
    config = parse_data_config(data_config)
    train_path = config['train']

    datasets = CustomDataset(train_path, 416, True, DEFAULT_TRANSFORM)
    dataloader = DataLoader(datasets, batch_size=2, collate_fn=datasets.collate_fn)
    data = next(iter(dataloader))
    print(data)


