import argparse
import os.path

import torch
import tqdm
from utils.tools import use_determinism
from model import load_model
import torch.optim as optim
from utils.parse_config import parse_data_config
from datasets import CustomDataset
from torch.utils.data import DataLoader
from utils.transform import DEFAULT_TRANSFORM
from utils.loss import compute_loss
import torch.utils.tensorboard as tensorboard
from torchsummary import summary
from terminaltables import AsciiTable
import datetime

def create_dataset(train_path, image_size, mutiscale, batch_size, n_cpu):
    dataset = CustomDataset(train_path,
                            imgsize=image_size,
                            mutiscale=mutiscale,
                            transform=DEFAULT_TRANSFORM)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=dataset.collate_fn,
                            pin_memory=False,
                            num_workers=n_cpu)
    return dataloader


def create_valid_dataset(valid_path, image_size, batch_size, n_cpu):
    dataset = CustomDataset(valid_path,
                            imgsize=image_size,
                            transform=DEFAULT_TRANSFORM)
    dataloader = DataLoader(dataset,
                            batch_size,
                            collate_fn=dataset.collate_fn,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=n_cpu)
    return dataloader


def run():
    parser = argparse.ArgumentParser(description=" Trains the YOLO model.")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to the model definition")
    parser.add_argument("-p", "--pretrained_weights", type=str, default="weights/darknet53_conv74.pth",
                        help="Path to the pretrained model weights")
    parser.add_argument("-e", "--epoch", type=int, default=50, help="definite the training epochs")
    parser.add_argument("--valid_interval", type=int, default=1, help="epoch interval for once evaluation")
    parser.add_argument("--log_dir", type=str, default="logs", help="log dir for tensorboard ")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of workers in dataloader")
    parser.add_argument('--mutiscale', type=bool, default=False, help=" if use mutiscaled images for training")
    parser.add_argument('--checkpoint_epoch_interval', type=int, default=1, help="interval of epoch to save weights")
    parser.add_argument('--seed', type=int, default=-1, help="make result reproducible")
    parser.add_argument('--verbose', action="store_true", help="show more verbose")
    args = parser.parse_args()

    # seed for cudnn, pytorch, numpy, python
    if args.seed != -1:
        use_determinism(args.seed)
    # tensorboard
    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = tensorboard.SummaryWriter(log_dir)

    # TODO: parse the data config file
    data_config = parse_data_config(args.data)
    train_path = data_config['train']
    val_path = data_config['valid']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: build the YOLOv3 model --> model.hyperparams: yolov3.cfg [net], optimizer = args.optim

    model = load_model(args.model, args.pretrained_weights).to(device)
    if args.verbose:
        summary(model, input_size=(model.hyper_params["channels"], model.hyper_params["height"],
                                   model.hyper_params["width"]), batch_size=1, device="cuda")

    # TODO: create the dataloader for training and validation
    mini_batch = model.hyper_params['batch'] // model.hyper_params['subdivisions']
    train_dataloader = create_dataset(train_path,
                                      model.hyper_params['width'],
                                      args.mutiscale,
                                      mini_batch,
                                      args.n_cpu)

    valid_dataloader = create_valid_dataset(val_path,
                                            model.hyper_params['width'],
                                            mini_batch,
                                            args.n_cpu)

    # TODO: definite optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if model.hyper_params['optimizer'] in ['Adam', None]:
        optimizer = optim.Adam(params,
                               lr=model.hyper_params['learning_rate'],
                               weight_decay=model.hyper_params['decay'])
    elif model.hyper_params['optimizer'] == 'sgd':
        optimizer = optim.SGD(params,
                              lr=model.hyper_params['learning_rate'],
                              weight_decay=model.hyper_params['decay'],
                              momentum=model.hyper_params['momentum'])

    # TODO: training phase
    model.train()
    print("\n---model training--\n")
    for epoch in range(args.epoch):
        for batch_i, (img_path, images, targets) in enumerate(tqdm.tqdm(train_dataloader,
                                                                        desc=f"training epoch {epoch}")):
            batch_done = batch_i + epoch * len(train_dataloader)
            images = images.to(device)
            outputs = model(images)
            targets = targets.to(device)
            loss, loss_summary = compute_loss(outputs, targets, model)
            loss.backward()

            l_box, l_obj, l_cls, loss = loss_summary
            writer.add_scalar("l_box", l_box.item(), batch_done)
            writer.add_scalar("l_obj", l_obj.item(), batch_done)
            writer.add_scalar("l_cls", l_cls.item(), batch_done)
            writer.add_scalar("loss", loss.item(), batch_done)

            lr_steps = model.hyper_params["lr_steps"]
            lr = model.hyper_params["learning_rate"]
            burn_in = model.hyper_params["burn_in"]
            if batch_done < burn_in:
                lr *= (batch_done / burn_in)
            else:
                for lr_scale, steps in lr_steps:
                    if batch_done > steps:
                        lr *= lr_scale
            optimizer.param_groups[0]["lr"] = lr

            writer.add_scalar("learning rate", lr, batch_done)

            if batch_done % mini_batch == 0:
                optimizer.step()
                optimizer.zero_grad()

        if epoch % args.checkpoint_epoch_interval == 0:
            checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            model.save_weights(checkpoint_path)

        if args.verbose:
            table = [["Type", "loss"],
                     ["loss of iou", round(l_box.item(), 6)],
                     ["loss of obj", round(l_obj.item(), 6)],
                     ["loss of class", round(l_cls.item(), 6)],
                     ["total loss", round(loss.item(), 6)]]

            print(AsciiTable(table).table)

    # TODO: validation phase


run()
