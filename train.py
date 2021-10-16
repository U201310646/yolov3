import argparse
import torch
import torch.optim as optim
from utils.parse_config import parse_data_config
def run():
    parser = argparse.ArgumentParser(description=" Trains the YOLO model.")
    parser.add_argument("-d", "-data", type=str, default="config/coco.data", help="Path to data config file")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to the model definition")
    parser.add_argument("-p", "--pretrained_weights", type=str, default="weights/yolov3.weights",
                        help="Path to the pretrained model weights")
    parser.add_argument("-e","--epoch", type=int, default=50, help="definite the training epochs")
    parser.add_argument("--valid_interval", type=int, default=1, help="epoch interval for once evaluation")
    parser.add_argument("--logdir", type=str, default="logs", help="logdir for tensorboard ")
    args = parser.parse_args()

    # TODO: parse the data config file
    data_config = parse_data_config(args.data)
    train_path = data_config['train']
    val_path = data_config['valid']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO: build the YOLOv3 model --> model.hyperparams: yolov3.cfg [net], optimizer = args.optim

    model = load_model(args.model, args.pretrained_weights)

    # TODO: create the dataloader for training and validation
    """
    train_dataloder = create_dataset(train_path)
    valid_dataloader = create_valid_dataset(valid_path)
    """
    # TODO: definite optimizer
    """
    if mdoel.hyperparams["optimizer"] in ["Adam",None]:
        optimizer = optim.Adam(params,lr,weight_decay)
    else:
        optimizer = optim.SGD(params,lr,weight_decay,momentum)
    """
    mini_batch = model.hyper_params['batch'] // model.hyper_params['subdivisions']
    # TODO: training phase
   """
   for epoch in range(args.epoch):
        for batch, (imgs, targets) in enumerate(train_dataloader):
            batch_done = batch + len(train_dataloader)*epoch
            outputs = model(imgs)
            loss = compute_loss(outputs, targets)
            loss.backward()
            if batch_done % mini_batch == 0:
                optimizer.step()
                optimizer.zero_grad()
   """
    #   TODO: validation phase






