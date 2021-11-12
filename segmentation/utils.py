import os
import torch
import numpy as np
import torch.nn as nn
import requests
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

class DuckietownTrainDataset(Dataset):
    def __init__(self, image_dir, mask_dir, real_image_dir="", real_mask_dir="", transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images_name = os.listdir(image_dir)  
        self.images_data = []
        for name in self.images_name:
            img_path = os.path.join(self.image_dir, name)
            if not "Real" in img_path:
                mask_path = os.path.join(self.mask_dir, name.replace("real", "result"))
            else:
                mask_path = os.path.join(self.mask_dir, name.replace("Real", "Segment"))    
                
            #0 - black - [road]
            #1 - white - [roadside]
            #2 - yellow - [markup]
            #3 - red - [crossroads]
            #4 - lilac -[duck]  
            mask = np.array(Image.open(mask_path).convert("P"))
            #mask = mask.numpy()
            for i in [0, 1, 2, 3, 4]:
                mask[mask==i] = 0    
            for i in [182, 189, 218, 219, 224, 225]:
                mask[mask==i] = 1
            for i in [38, 39, 44, 45, 74, 80, 81, 110, 117, 146, 153]:
                mask[mask==i] = 2
            for i in [11, 12, 13, 14, 15, 19, 20, 21, 49, 50, 51, 55, 56, 57]:
                mask[mask==i] = 3
            #for i in [167, 168, 174, 175, 204, 209, 210, 211]:
            #    mask[mask==i] = 4
            mask[mask>=4] = 0   
            
            image = Image.open(img_path).convert("RGB")
            '''
            #save basic image
            if self.transform is not None:
                augmentations = self.transform(image=np.array(image), mask=mask)
                image2 = augmentations["image"]
                mask2 = augmentations["mask"]
                self.images_data.append([image2, mask2])
            '''
            #augmentation BLUR
            image1 = image.filter(ImageFilter.BLUR)
            image1 = np.array(image1)
            if self.transform is not None:
                augmentations = self.transform(image=image1, mask=mask)
                image1 = augmentations["image"]
                mask1 = augmentations["mask"]
                self.images_data.append([image1, mask1])    
            '''
            
            #augmentation FLIP
            image1 = image.transpose(Image.FLIP_LEFT_RIGHT)
            image1 = np.array(image1)
            if self.transform is not None:
                augmentations = self.transform(image=image1, mask=mask)
                image1 = augmentations["image"]
                mask1 = augmentations["mask"]
                self.images_data.append([image1, mask1])   
            
            #augmentation BLUR and FLIP
            image2 = image.filter(ImageFilter.BLUR).transpose(Image.FLIP_LEFT_RIGHT)
            image2 = np.array(image2)
            if self.transform is not None:
                augmentations = self.transform(image=image2, mask=mask)
                image1 = augmentations["image"]
                mask2 = augmentations["mask"]
                self.images_data.append([image1, mask2])    
            '''

        #Download real image    
        self.real_image_dir = real_image_dir
        self.real_mask_dir = real_mask_dir
        if self.real_image_dir and self.real_mask_dir:
            self.real_images_name = os.listdir(self.real_image_dir)  
            for name in self.real_images_name:
                img_path = os.path.join(self.real_image_dir, name)
                mask_path = os.path.join(self.real_mask_dir, name.replace("jpg", "npy"))  
                mask = np.load(mask_path)
                image = Image.open(img_path).convert("RGB")
                #save basic image
                if self.transform is not None:
                    augmentations = self.transform(image=np.array(image), mask=mask)
                    image2 = augmentations["image"]
                    mask2 = augmentations["mask"]
                    self.images_data.append([image2, mask2])
            '''  
            #augmentation FLIP  
            image1 = image.transpose(Image.FLIP_LEFT_RIGHT)
            image1 = np.array(image1)
            if self.transform is not None:
                augmentations = self.transform(image=image1, mask=mask)
                image1 = augmentations["image"]
                mask1 = augmentations["mask"]
                self.images_data.append([image1, mask1]) 
            '''   

    def __len__(self):
        return len(self.images_data)
 
    def __getitem__(self, index): 
        return self.images_data[index][0], self.images_data[index][1]

def save_checkpoint(state, dir_name, filename):
    print(f"=> Saving checkpoint {filename}")
    torch.save(state, f"{dir_name}{filename}")
 
def load_checkpoint(checkpoint, model, device="cpu"):
    if device == "cpu":
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint)   
    print(f"=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def load_checkpoint_from_url(url, model, device="cpu"):
    if device == "cpu":
        checkpoint = torch.load(requests.get(url).json(), map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(requests.get(url).json())   
    print(f"=> Loading checkpoint from url")
    model.load_state_dict(checkpoint["state_dict"])    
    
#Datasets loaders
def get_train_loaders(
    train_dir,
    train_maskdir,
    real_real,
    real_mask,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers,
    pin_memory=True,
):
    train_ds = DuckietownTrainDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        real_image_dir=real_real,
        real_mask_dir=real_mask,
        transform=train_transform,
    )
 
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
 
    val_ds = DuckietownTrainDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )
 
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
 
    return train_loader, val_loader
    
#Check training accuracy
def check_accuracy(loader, model, device="cpu"):
    print('Check accuracy')
    num_correct = 0
    num_pixels = 0
    dice_scoreAll = 0
    smooth = 1e-3
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.cpu().numpy()
            dice_score = 0
            some = model(x)
            preds = nn.functional.softmax(some, dim=1)
            preds = torch.argmax(preds, dim=1, keepdim=False).cpu().numpy()
            
            #for background
            back = preds.copy()
            back[back > 0] = 5 # change not background
            back_num = (back == 0).sum()
            back_mask_num = (y == 0).sum()
            num_correct = (back == y).sum()
            dice_score += 2 * (num_correct + smooth)/(back_num + back_mask_num + smooth)

            #for roadside
            roadside = preds.copy()
            roadside[roadside > 1] = 5 # change not roadside
            roadside[roadside < 1] = 5 # change not roadside
            roadside_num = roadside[roadside == 1].sum()
            roadside_mask_num = y[y == 1].sum()
            num_correct = (roadside == y).sum()
            dice_score += 2 * (num_correct + smooth)/(roadside_num + roadside_mask_num + smooth)
            
            #for markup
            markup = preds.copy()
            markup[markup > 2] = 5 # change not markup
            markup[markup < 2] = 5 # change not markup
            markup_num = markup[markup == 2].sum()
            markup_mask_num = y[y == 2].sum()
            num_correct = (markup == y).sum()
            dice_score += 2 * (num_correct + smooth)/(markup_num + markup_mask_num + smooth)
            
            #for crossroads
            crossroads = preds.copy()
            crossroads[crossroads > 3] = 5 # change not crossroads
            crossroads[crossroads < 3] = 5 # change not crossroads
            crossroads_num = crossroads[crossroads == 3].sum()
            crossroads_mask_num = y[y == 3].sum()
            num_correct = (crossroads == y).sum()
            dice_score += 2 * (num_correct + smooth)/(crossroads_num + crossroads_mask_num + smooth)
            
            dice_scoreAll += dice_score / 4
            
    print(f"Dice score: {dice_scoreAll/len(loader)}")  
    return dice_scoreAll/len(loader)

#Predict classes with model
def get_predict(
    idx, image_to_change, model, 
    device="cpu"
):
    model.eval()
    #print(f"Get preds {idx}") 
    x = image_to_change.to(device=device)
    x = torch.unsqueeze(x, 0)
    with torch.no_grad():
        preds = model(x.float())
        preds = nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1, keepdim=False)
        return preds 
    
def get_predict_with_time(
    idx, image_to_change, model, 
    device="cuda"
):
    model.eval()
    #print(f"Get preds {idx}") 
    x = image_to_change.to(device=device)
    x = torch.unsqueeze(x, 0)
    with torch.autograd.profiler.profile(use_cuda=False) as prof1:
        preds = model(x.float())
        preds = nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1, keepdim=False)
    time = prof1.self_cpu_time_total/1000 
    print("original model: {:.2f}ms".format(time))
    return preds, time
    
def get_predict_img_with_time(
    idx, image_to_change, model, 
    device="cuda"
):

    model.eval()
    x = image_to_change.to(device=device)
    x = torch.unsqueeze(x, 0)
    with torch.autograd.profiler.profile(use_cuda=False) as prof1:
        preds = model(x.float())
        preds = nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1, keepdim=False)
        preds = simple_save_preds_image_from_tensor(preds)
    time = prof1.self_cpu_time_total/1000 
    print("original model: {:.2f}ms".format(time))
    return preds, time    
        
#Saving segment picture  
def save_im(arr, idx, folder, width, height):
    #------CONVERT "P-mode" tenzor IN IMAGE--------
 
    #0 - black - [road]
    #1 - white - [roadside]
    #2 - yellow - [markup]
    #3 - red - [crossroads]
    #4 - lilac -[duck]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            elem = arr[i][j]
            if elem == 1:
                image[i][j] = [255, 255, 255]
            elif elem == 2:
                image[i][j] = [255, 255, 0]
            elif elem == 3:
                image[i][j] = [255, 0, 0]
            #elif elem == 4:
            #    image[i][j] = [102, 114, 232]       
    image = Image.fromarray(image, 'RGB')
    save_path = os.path.join(folder, f"result{idx}.jpeg")
    image.save(save_path)      
 
def save_preds_image_from_tensor(tenz, idx, folder, width, height):
     #------CONVERT "P-mode" tenzor IN IMAGE--------
 
    #0 - black - [road]
    #1 - white - [roadside]
    #2 - yellow - [markup]
    #3 - red - [crossroads]
    #4 - lilac -[duck]
 
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            #print(tenz.shape)
            elem = tenz[0][i][j]
            if elem == 1:
                image[i][j] = [255, 255, 255]
            elif elem == 2:
                image[i][j] = [255, 255, 0]
            elif elem == 3:
                image[i][j] = [255, 0, 0]
            #elif elem == 4:
            #    image[i][j] = [102, 114, 232]      
    image = Image.fromarray(image, 'RGB')
    save_path = os.path.join(folder, f"result{idx}.jpeg")
    image.save(save_path) 
    
def simple_save_preds_image_from_tensor(preds, idx=0, folder="./result1", save_flag=False):
    image = np.dstack([preds[0], preds[0], preds[0]])
    image = (image * 100 % 255).astype(np.uint8)
    if save_flag:
        image = Image.fromarray(image, 'RGB')
        if not os.path.exists(folder):
            os.makedirs(folder) 
        save_path = os.path.join(folder, f"result{idx}.png")
        image.save(save_path)      
        image = np.array(image)
    return image     
 
def save_predictions_as_two_images(
    idx, preds, r_image, model, 
    folder_real, 
    folder_seg, 
    width, 
    height,
    device="cpu"
):
    save_preds_image_from_tensor(preds, idx, folder_seg, width, height)
    r_img = Image.fromarray(r_image)
    save_path = os.path.join(folder_real, f"real{idx}.jpeg")
    r_img.save(save_path)  

def train_fn(loader, model, optimizer, loss_fn, scaler, device="cpu"):
    print("Train_fn")
    model.train()
    
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device).float()
        targets = targets.to(device).long()
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            predictions = nn.functional.softmax(predictions, dim=1)
            loss = loss_fn(predictions, targets)
 
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
   
        torch.cuda.empty_cache()
