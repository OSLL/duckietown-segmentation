import os
from PIL import Image
import numpy as np
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation import *

class EDANetWrapper:
    def __init__(self, model_path, num_classes="4", device="cpu"):
        self.device = device
        self.model = load_EDANet_model(model_path, num_classes, device)  
        
    def observation(self, obs):
        return get_predict(obs, self.model, self.device)

def run_segmentation(args):
    if not os.path.exists(args.test_dir):
        print(f"Test image directory {args.save_real_dir} does not exists")
        exit(1)
    if not os.path.exists(args.model_path):
        print(f"Model checkpoint named {args.model_path} does not exists")
        exit(1)        
 
        
    device = args.device    
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
    image_dir = args.test_dir
    images = os.listdir(image_dir)
    images_count = len(images)
    sum_time = 0
    
    #markup of pictures from a folder 
    for idx in range(images_count):
        print(f"Image index {idx}")
        img_path = os.path.join(image_dir, images[idx])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        image_r = image.copy()
        augmentations = test_transform(image=image)
        image = augmentations["image"]
        preds, time = get_predict_img_with_time(idx, image, model, device=device)
        #image.save(save_path) 
        print(time)
        print(preds)
        if idx >= 2:
            sum_time += time
    print(sum_time/(images_count - 2))
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", type=str)  # Set device
    parser.add_argument("--height", default=480, type=int) # Height input image
    parser.add_argument("--width", default=640, type=int) # Width input image
    parser.add_argument("--num_classes", default=4, type=int) # Number of classes in image after segmentation
    parser.add_argument("--test_dir", type=str) # Directory to start segmentation (./<name>)
    parser.add_argument("--model_path", type=str) # Directory to load a neural network checkpoint (./<name>)
    parser.add_argument("--net_name", default="edanet", type=str) # [edanet, dabnet, unet]
    run_segmentation(parser.parse_args())         
