# Installing
```
pip3 install git+https://github.com/OSLL/duckietown-segmentation.git@main
```
or
```
git clone https://github.com/OSLL/duckietown-segmentation.git
cd duckietown-segmentation
python3 -m pip install .
```
# Usage
```
from segmentation import <function_name>
```
# Example

```
python3
> import numpy as np
> import albumentations as A
> from albumentations.pytorch import ToTensorV2
> from segmentation import *
> model = load_DABNet_model("./segmentation/dabnetlr=0.0003optim=Adamepoch=39.pth.tar", 4, device='cpu')
> image = np.random.randint(0, 255, (480, 640, 3))
> transform = A.Compose(
         [
             ToTensorV2(),
         ],
     )
> augmentations = transform(image=image)
> image = augmentations["image"]
> preds = get_predict(0, image, model, device='cpu')
> save_predictions_as_two_images(0, preds, image_r.astype(np.uint8), model, './', './', 320, 240, 'cpu')
```
