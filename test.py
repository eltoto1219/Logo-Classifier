import os
from matplotlib import pyplot as plt
import PIL
import numpy as np

image_path = "/home/antonio/Desktop/trill/images/pngs"
files = [os.path.join(image_path, x) for x in os.listdir(image_path)]

image = PIL.Image.open(files[1])
image = image.convert("RGB")
image = np.asarray(image, dtype=np.float32) / 255
image = image[:, :, :3]

# plt.imshow(image)
# plt.show()
for x in files:
	image = PIL.Image.open(files[1])
	image = image.convert("RGB")
	image = np.asarray(image, dtype=np.float32) / 255
	image = image[:, :, :3]
	print(image.shape)

