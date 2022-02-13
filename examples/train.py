import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation import *


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

def run_train(args):

    train_transform = A.Compose(
        [
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2(),
        ],
    )

    if not os.path.exists(args.train_sim_images_r_dir):
        print(f"Train simulator real image directory {args.train_sim_images_r_dir} does not exists")
        exit(1)
    if not os.path.exists(args.train_sim_images_m_dir):
        print(f"Train simulator mask image directory {args.train_sim_images_m_dir} does not exists")
        exit(1)     
    if not os.path.exists(args.test_images_r_dir):
        print(f"Test real image directory {args.test_images_r_dir} does not exists")
        exit(1)
    if not os.path.exists(args.test_images_m_dir):
        print(f"Test mask image directory {args.test_images_m_dir} does not exists")
        exit(1)   
    if args.train_real_images_r_dir and not os.path.exists(args.train_real_images_r_dir):
        print(f"Train real image directory {args.train_real_images_r_dir} does not exists")
        exit(1)
    if args.train_real_images_m_dir and not os.path.exists(args.train_real_images_m_dir):
        print(f"Traint real image directory {args.train_real_images_m_dir} does not exists")   
        exit(1)    
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)      
    if args.net_name == "edanet":
        model = EDANet(args.num_classes).to(args.device)
    elif args.net_name == "dabnet":
        model = DABNet(args.num_classes).to(args.device)
    elif args.net_name == "unet":
        model = UNet(args.num_classes).to(args.device)

    if args.lr <= 0:
        print("The liarning must be positive float number. Now is {args.num_epochs}")
        exit(1)
        
    loss_fn = nn.CrossEntropyLoss()
    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0001)
    elif args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)    

    train_loader, val_loader = get_train_loaders(
        args.train_sim_images_r_dir,
        args.train_sim_images_m_dir,
        args.train_real_images_r_dir,
        args.train_real_images_m_dir,
        args.test_images_r_dir,
        args.test_images_m_dir,
        args.batch_size,
        train_transform,
        val_transforms,
        args.num_workers,
    )

    '''
    if LOAD_MODEL:
        load_checkpoint(torch.load(f"{LOAD_MODEL_DIR}my_checkpoint1.pth.tar"), model)
    '''

    scaler = torch.cuda.amp.GradScaler()

    if args.num_epochs <= 0:
        print("The number of epochs for learning must be positive number. Now is {args.num_epochs}")
        exit(1)
    for epoch in range(args.num_epochs):
        print(f"Train epoch {epoch}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, args.device)

        # save model
        checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, args.save_model_dir, f"{args.save_model_name}lr={args.lr}optim={args.optim}epoch={epoch}.pth.tar")

        # check accuracy
        check_accuracy(val_loader, model, device=args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", type=str)  # Set device [cpu, cuda]
    parser.add_argument("--height", default=480, type=int) # Height input image
    parser.add_argument("--width", default=640, type=int) # Width input image
    parser.add_argument("--num_classes", default=4, type=int) # Number of classes in image after segmentation
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--lr", default=4.5e-2, type=float) # Learning rate for train
    parser.add_argument("--batch_size", default=10, type=int) # Batch size for train
    parser.add_argument("--num_epochs", default=15, type=int) # Number of epochs for training
    parser.add_argument("--optim", default="sgd", type=str) # [sgd, adam]
    parser.add_argument("--train_sim_images_r_dir", type=str) # Directory with train simulator real images(./<name>)
    parser.add_argument("--train_sim_images_m_dir", type=str) # Directory with train simulator mask images(./<name>)
    parser.add_argument("--train_real_images_r_dir", type=str) # Directory with train images from real world(./<name>)
    parser.add_argument("--train_real_images_m_dir", type=str) # Directory with train mask images from real world(./<name>)
    parser.add_argument("--test_images_r_dir", type=str) # Directory to load real images, which check accuracy (./<name>)
    parser.add_argument("--test_images_m_dir", type=str) # Directory to load segment images, which check accuracy (./<name>)
    parser.add_argument("--save_model_dir", default="./checkpoints", type=str) # Directory for saving checkpoint (./<name>)
    parser.add_argument("--save_model_name", default="checkpoint", type=str) # Name for saving checkpoint (<name>)
    parser.add_argument("--net_name", default="edanet", type=str) # [edanet, dabnet, unet]
    run_train(parser.parse_args())             
