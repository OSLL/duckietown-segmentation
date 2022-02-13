import os
from PIL import Image
import numpy as np

img_path = os.path.join("./test-pictures", "Segment-Duckiebot-605.jpeg")
image = Image.open(img_path).convert("P")
mask = np.array(image)
#print(image.shape)
#for i in range(600):
#    for j in range(800):
#        print(i, j, mask[i][j])

for i in [0, 1, 2, 3, 4]:
    mask[mask==i] = 0    
for i in [182, 189, 218, 219, 224, 225]:
    mask[mask==i] = 1
for i in [38, 39, 44, 45, 74, 80, 81, 110, 117, 146, 153]:
    mask[mask==i] = 2
for i in [12, 13, 14, 15, 19, 20, 21, 49, 50, 51, 55, 56, 57]:
    mask[mask==i] = 3 
for i in [137, 142, 143, 178, 179]:
    mask[mask==i] = 4
mask[mask>4] = 0

image = np.dstack([mask, mask, mask])
image = (image * 100 % 255).astype(np.uint8)
image = Image.fromarray(image, 'RGB')

#image = image.resize((84, 84))
save_path = os.path.join("./test-pictures", f"result1.png")
image.save(save_path)     
