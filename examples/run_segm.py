import os
from PIL import Image
import numpy as np
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation import *

def run_segmentation(args):
    if not os.path.exists(args.test_dir):
        print(f"Test image directory {args.save_real_dir} does not exists")
        exit(1)
    if not os.path.exists(args.model_path):
        print(f"Model checkpoint named {args.model_path} does not exists")
        exit(1)        
    if not os.path.exists(args.save_real_dir):
        os.makedirs(args.save_real_dir)     
    if not os.path.exists(args.save_segm_dir):
        os.makedirs(args.save_segm_dir) 
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

    #markup of pictures from a folder 
    for idx in range(images_count):
        print(f"Image index {idx}")
        img_path = os.path.join(image_dir, images[idx])
        image = Image.open(img_path).convert("RGB")
        #image = image.crop((0, 160, 640, 480)) # cut the top of image
        image = np.array(image)
        print(image.shape)
        image_r = image.copy()
        augmentations = test_transform(image=image)
        image = augmentations["image"]
        print(image.shape)
        preds = get_predict(idx, image, model, device=device)
        image = simple_save_preds_image_from_tensor(preds, idx, save_flag=False)
        #save_predictions_as_two_images(idx, preds, image_r, model, args.save_real_dir, args.save_segm_dir, width=args.width, height=args.height, device=device)
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", type=str)  # Set device
    parser.add_argument("--height", default=480, type=int) # Height input image
    parser.add_argument("--width", default=640, type=int) # Width input image
    parser.add_argument("--num_classes", default=4, type=int) # Number of classes in image after segmentation
    parser.add_argument("--test_dir", type=str) # Directory to start segmentation (./<name>)
    parser.add_argument("--save_real_dir", type=str) # Directory to save real image for segmentation (./<name>)
    parser.add_argument("--save_segm_dir", type=str) # Directory to save segmentation image (./<name>)
    parser.add_argument("--model_path", type=str) # Directory to load a neural network checkpoint (./<name>)
    parser.add_argument("--net_name", default="edanet", type=str) # [edanet, dabnet, unet]
    run_segmentation(parser.parse_args())         
