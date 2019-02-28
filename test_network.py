from image_eval import *
from logo_nn import *
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as d
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from run import *
import os
import PIL



view = iter(train_loader)
view = view.next()

image = view[0][0]
label = view[1][0]

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title("This  has a label of {}".format(label))
    plt.show()

#imshow(image)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logoNet = Net().to(dev)

checkpoint = torch.load("/home/antonio/Desktop/trill/project1/chkpnts/model_chkpnt_epoch_70_.tar", map_location = "cpu")
logoNet.load_state_dict(checkpoint['model_state_dict'])
logoNet.eval()

image_path = "/home/antonio/Desktop/trill/images/pngs"
files = [os.path.join(image_path, x) for x in os.listdir(image_path)]
num_logos = len(files)

trill_logos = []

trans = [transforms.ToPILImage(), transforms.Pad(10), 
    transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))]

# for x in files:
# 	image = PIL.Image.open(x)
# 	image = image.convert("RGB")
# 	image = np.asarray(image, dtype=np.float32) / 255
# 	image = image[:, :, :3]
# 	image = torch.from_numpy(image)
# 	for j in trans:
# 		z = j(image)
#	trill_logos.append(y)
for i, x in enumerate(files):
	image = PIL.Image.open(x)
	image = image.convert("RGB")
	image = np.asarray(image, dtype=np.float32) / 255
	image = image[:, :, :3]
	#image = torch.from_numpy(image)
	test_im = torch.from_numpy(image)
	for x in trans:
		test_im = x(test_im)
	test_im = test_im.unsqueeze(0)
	trill_logos.append(test_im)


for x in trill_logos:
	outputs = logoNet(x)
	_, predicted = torch.max(outputs, 1)
	classes = ["logo", "random"]
	print("predicted: {}".format(outputs))


not_logo = "/home/antonio/Desktop/trill/images/pngs/ABN Newswire (Chinese - Simplified).png"
not_logo = PIL.Image.open(not_logo)
not_logo = not_logo.convert("RGB")
not_logo = np.asarray(not_logo, dtype=np.float32) / 255
not_logo = not_logo[:, :, :3]
not_logo = torch.from_numpy(not_logo)
for x in trans:
	not_logo = x(not_logo)
not_logo = not_logo.unsqueeze(0)

print()

outputs = logoNet(not_logo)
_, predicted = torch.max(outputs, 1)
classes = ["logo", "random"]
print("predicted for not logo: {}".format(outputs))