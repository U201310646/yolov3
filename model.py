import os

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.parse_config import parse_model_config


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Yololayer(nn.Module):
    def __init__(self, anchors):
        super(Yololayer, self).__init__()
        self.register_buffer('anchors', torch.tensor(anchors))
        self.stride = None
        self.wh = None

    def forward(self, x, img_size):
        na = len(self.anchors)
        b, _, nx, ny = x.shape
        self.stride = img_size // nx
        self.wh = nx
        # [N,255,W,H] --> [N,3,W,H,85]
        x = x.view(b, na, -1, nx, ny).permute(0, 1, 4, 3, 2).contiguous()

        if not self.training:
            grid = self._mesh_grid(nx, ny).to(x.device)
            # image coordinate
            x[..., :2] = (torch.sigmoid(x[..., :2]) + grid) * self.stride
            anchor_wh = self.anchors.clone().view(1, -1, 1, 1, 2)
            x[..., 2:4] = torch.exp(x[..., 2:4]) * anchor_wh
            x[..., 4:] = x[..., 4:].sigmoid()
            x = x.view(b, -1, x.shape[-1])
        return x

    @staticmethod
    def _mesh_grid(nx, ny):
        return torch.stack(torch.meshgrid(torch.arange(nx), torch.arange(ny)), 2).view(1, 1, nx, ny, 2)


def weight_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0, 1)
        # no bias if batch_normalization
        # nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 0, 1)
        nn.init.constant_(module.bias, 0)


def create_modules(model_defs):
    """
    :param model_defs: model definition from model.cfg [{},{},..]
    :return: (hyper_params, model_list)
            model_list: nn.ModuleList += [every layer create a Seqential]
    """
    hyper_params = model_defs.pop(0)
    hyper_params.update({
        'batch': int(hyper_params.get('batch')),
        'subdivisions': int(hyper_params.get('subdivisions')),
        'width': int(hyper_params.get('width')),
        'height': int(hyper_params.get('height')),
        'channels': int(hyper_params.get('channels')),
        'momentum': float(hyper_params.get('momentum')),
        'decay': float(hyper_params.get('decay')),
        'angle': float(hyper_params.get('angle')),
        'saturation': float(hyper_params.get('saturation')),
        'exposure': float(hyper_params.get('exposure')),
        'hue': float(hyper_params.get('hue')),
        'learning_rate': float(hyper_params.get('learning_rate')),
        'burn_in': int(hyper_params.get('burn_in')),
        'max_batches': int(hyper_params.get('max_batches')),
        'policy': hyper_params.get('policy'),
        'optimizer': hyper_params.get('optimizer'),
        'lr_steps': list(zip(map(float, hyper_params.get('scales').split(',')),
                             map(int, hyper_params.get('steps').split(','))))
    })
    model_list = nn.ModuleList()
    output_filters = [int(hyper_params['channels'])]

    for model_i, model_def in enumerate(model_defs):
        seq_model = nn.Sequential()
        if model_def['type'] == 'convolutional':
            filters = int(model_def['filters'])
            kernel = int(model_def['size'])
            stride = int(model_def['stride'])
            pad = (kernel - stride + 1) // 2
            activation = model_def['activation']
            batch_normalization = int(model_def.get('batch_normalize', 0))

            seq_model.add_module(
                f'conv_{model_i}',
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel,
                    stride=stride,
                    padding=pad,
                    bias=not batch_normalization))

            if batch_normalization:
                seq_model.add_module(f"bn_{model_i}", nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
            if activation == "leaky":
                seq_model.add_module(f"leaky_{model_i}", nn.LeakyReLU(0.1))
            if activation == "mish":
                seq_model.add_module(f"mish_{model_i}", Mish())
        elif model_def['type'] == 'upsample':
            seq_model.add_module(f"upsample_{model_i}", Upsample(scale_factor=int(model_def['stride']), mode='nearest'))
        elif model_def['type'] == 'shortcut':
            filters = output_filters[int(model_def['from'])]
            seq_model.add_module(f"shortcut{model_i}", nn.Sequential())
        elif model_def['type'] == 'route':
            filters = sum([output_filters[int(layer)] for layer in model_def['layers'].split(',')])
            seq_model.add_module(f"route_{model_i}", nn.Sequential())
        elif model_def['type'] == 'yolo':
            anchor = [int(anchor_i.strip()) for anchor_i in model_def['anchors'].split(',')]
            anchor_num = int(model_def['num'])
            anchors = [(anchor[2 * i], anchor[2 * i + 1]) for i in range(anchor_num)]
            mask = [int(mask.strip()) for mask in model_def['mask'].split(',')]
            anchors = [anchors[i] for i in mask]
            seq_model.add_module(f"yolo_{model_i}", Yololayer(anchors))

        output_filters.append(filters)
        model_list.append(seq_model)
    return hyper_params, model_list


class Darknet(nn.Module):
    def __init__(self, model_config):
        super(Darknet, self).__init__()
        self.model_defs = parse_model_config(model_config)
        self.hyper_params, self.model_list = create_modules(self.model_defs)
        self.yolo_layer = [yolo[0] for yolo in self.model_list if isinstance(yolo[0], Yololayer)]

    def forward(self, x):
        yolo_outputs = []
        layer_outputs = []
        img_size = x.shape[2]
        for model_def, module in zip(self.model_defs, self.model_list):
            if model_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif model_def['type'] == 'shortcut':
                x = layer_outputs[int(model_def['from'])] + layer_outputs[-1]
            elif model_def['type'] == 'route':
                x = torch.cat([layer_outputs[int(layer)] for layer in model_def['layers'].split(',')], 1)
            elif model_def['type'] == 'yolo':
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            layer_outputs.append(x)

        return yolo_outputs

    def save_weights(self, checkpoint_path):
        ckt_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(ckt_dir):
            os.mkdir(ckt_dir)
        torch.save(self.state_dict(), checkpoint_path)

    def load_weights(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))


def load_model(model_config, weight_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Darknet(model_config).to(device)
    model.apply(weight_init)

    if weight_path:
        model.load_weights(weight_path)
    return model


def run():
    import cv2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Darknet(model_config='config/yolov3.cfg').to(device)
    model.eval()
    img_path = 'data/samples/dog.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (416, 416))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    # img shape -- [B,C,H,W]
    out = model(img)
    print(out)
