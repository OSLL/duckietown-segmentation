import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation import load_EDANet_model
from segmentation import load_UNet_model
from segmentation import load_DABNet_model
from segmentation import save_predictions_as_two_images, get_predict


class EDANetWrapper:
    def __init__(self, model_path, num_classes="4", height=480, width=640, device="cpu"):
        self.device = device
        self.model = load_EDANet_model(model_path, num_classes, device) 
        self.transform = A.Compose(
        [
            A.Resize(height=height, width=width),
            ToTensorV2(),
        ], )
        
    def segmentation(self, obs):
        augmentations = self.transform(image=obs)
        obs = augmentations["image"]
        return get_predict(0, obs, self.model, self.device)

class DABNetWrapper:
    def __init__(self, model_path, num_classes="4", height=480, width=640, device="cpu"):
        self.device = device
        self.model = load_DABNet_model(model_path, num_classes, device) 
        self.transform = A.Compose(
        [
            A.Resize(height=height, width=width),
            ToTensorV2(),
        ], )
        
    def segmentation(self, obs):
        augmentations = self.transform(image=obs)
        obs = augmentations["image"]
        return get_predict(0, obs, self.model, self.device)
    
class UNetWrapper:
    def __init__(self, model_path, num_classes="4", height=480, width=640, device="cpu"):
        self.device = device
        self.model = load_UNet_model(model_path, num_classes, device) 
        self.transform = A.Compose(
        [
            A.Resize(height=height, width=width),
            ToTensorV2(),
        ], )
        
    def segmentation(self, obs):
        augmentations = self.transform(image=obs)
        obs = augmentations["image"]
        return get_predict(0, obs, self.model, self.device)    
