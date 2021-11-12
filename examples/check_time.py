import os
from PIL import Image
import numpy as np
import argparse
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation import *

def check_time(args):      
    if torch.cuda.is_available() and args.device == "cuda":
        device = "cuda"
    else:
        device = "cpu"        
    if args.net_name == "edanet":
        model = load_EDANet_model(args.model_path, args.num_classes, device=device)  
    elif args.net_name == "dabnet":
        model = load_DABNet_model(args.model_path, args.num_classes, device=device) 
    elif args.net_name == "unet":
        model = load_UNet_model(args.model_path, args.num_classes, device=device) 

    test_transform = A.Compose(
        [
            A.Resize(height=args.height, width=args.width),
            ToTensorV2(),
        ],
    )
    
    image = np.array(Image.open("./test-pictures/test1.jpeg").convert("RGB"))
    augmentations = test_transform(image=image)
    image = augmentations["image"]
    preds, time = get_predict_img_with_time(0, image, model, device=device)
    #image = Image.fromarray(preds, 'RGB')
    #save_path = os.path.join("./", f"result.jpeg")
    #image.save(save_path) 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", type=str)  # Set device
    parser.add_argument("--height", default=480, type=int) # Height input image
    parser.add_argument("--width", default=640, type=int) # Width input image
    parser.add_argument("--num_classes", default=4, type=int) # Number of classes in image after segmentation
    parser.add_argument("--model_path", type=str) # Directory to load a neural network checkpoint (./<name>)
    parser.add_argument("--net_name", default="edanet", type=str) # [edanet, dabnet, unet]
    check_time(parser.parse_args())         
