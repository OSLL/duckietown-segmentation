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
from segmentation import EDANet, UNet, DABNet, get_train_loaders, train_fn, save_checkpoint, check_accuracy

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

import matplotlib.pyplot as plt

def print_plot(net_name, optim_name, lr, array_dices, array_y, path):
    plt.figure().clear()
    plt.plot(array_y, array_dices, label='Dice_score')
    plt.ylabel('Values')
    plt.xlabel('Epochs')
    plt.title(f"{net_name} lr={lr} optim={optim_name}")
    plt.legend()
    plt.savefig(path)
 
def run_train():

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2(),
        ],
    )
    device = "cuda"
    num_classes = 5
    num_workers = 1
    models_save_dir = "./models_new/"
    train_sim_images_r_dir = "./pictures/duckiebots_real/"
    train_sim_images_m_dir = "./pictures/duckiebots_segm/"
    train_real_images_r_dir = "./pictures/RealDuckiebots/"
    train_real_images_m_dir = "./pictures/SegmClassDuckiebots/"
    test_images_r_dir = "./pictures/test/real/"
    test_images_m_dir = "./pictures/test/segment/"
    
    list_models = ["edanet", "dabnet", "unet"]
    optim_list = ["RMSprop", "Adam", "AdamW",  "Adamax"]
    batch_size = 10
    num_epochs = 50
    lr_list = [
               1e-4, 3e-4, 5e-4, 7e-4,
               1e-3, 3e-3, 5e-3, 7e-3,
               1e-2, 3e-2, 5e-2]
        

    train_loader, val_loader = get_train_loaders(
                    train_sim_images_r_dir,
                    train_sim_images_m_dir,
                    train_real_images_r_dir,
                    train_real_images_m_dir,
                    test_images_r_dir,
                    test_images_m_dir,
                    batch_size,
                    train_transform,
                    val_transforms,
                    num_workers,
                )
    epochs_arr = []
    for i in range(0, num_epochs):
        epochs_arr.append(i)
    
    file = open("accuracy1.txt", "w")
    
    for lr in lr_list:
        print("------------", f"lr={lr}")
        file.write(f"------------ lr={lr}\n")
        for net_name in list_models:
            print("===============", f"model_name={net_name}")
            file.write(f"=============== model_name={net_name}\n")
            
            for optim_name in optim_list:
                print("*************", f"optim={optim_name}")
                file.write(f"************* optim={optim_name}\n")
                save_model_dir = f"{models_save_dir}{net_name}/{optim_name}/{lr}/"   
                
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                #if not os.path.exists(f"{save_model_dir}plots"):
                #    os.makedirs(f"{save_model_dir}plots")     
                
                if net_name == "edanet":
                    model = EDANet(num_classes).to(device)
                elif net_name == "dabnet":
                    model = DABNet(num_classes).to(device)
                elif net_name == "unet":
                    model = UNet(num_classes).to(device)
                    
                if optim_name == "RMSprop":
                    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.0001)
                elif optim_name == "Rprop":
                    optimizer = optim.Rprop(model.parameters(), lr=lr)     
                elif optim_name == "SGD":
                    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001)
                elif optim_name == "Adam":
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)   
                elif optim_name == "Adagrad":
                    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.0001) 
                elif optim_name == "AdamW":
                    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001) 
                elif optim_name == "Adamax":
                    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=0.0001)  
                elif optim_name == "ASGD":
                    optimizer = optim.ASGD(model.parameters(), lr=lr, weight_decay=0.0001)   
   

                scaler = torch.cuda.amp.GradScaler()
                loss_fn = nn.CrossEntropyLoss()
                array_dices = []
                for epoch in range(num_epochs):
                    print(f"Train epoch {epoch}")
                    train_fn(train_loader, model, optimizer, loss_fn, scaler, device)

                    # check accuracy
                    dice_score = check_accuracy(val_loader, model, device=device)  
                    
                    # save model
                    if dice_score > 0.6:
                        checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        }
                        save_checkpoint(checkpoint, save_model_dir, f"{net_name}lr={lr}optim={optim_name}epoch={epoch}D.pth.tar")
                    file.write(f"Epoch: {epoch} Dice score: {dice_score}\n")
                    array_dices.append(dice_score) 
                    
                print_plot(net_name, optim_name, lr, array_dices, epochs_arr, f"{models_save_dir}plots/{net_name} lr={lr}optim={optim_name}D.png")    
    file.close()
    
if __name__ == "__main__":
    run_train()             
