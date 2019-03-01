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



#loading network to specifc epoch to test accuracy
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logoNet = Net().to(dev)
checkpoint = torch.load("/home/antonio/Desktop/chkpnts/model_chkpnt_epoch_83_.tar", map_location = "cpu")
logoNet.load_state_dict(checkpoint['model_state_dict'])
logoNet.eval()


#seeting up dir for trill images and transforming them appropriately
image_path = "/home/antonio/Desktop/trill/trill_logos/logo"
files = [os.path.join(image_path, x) for x in os.listdir(image_path)]
num_logos = len(files)
trill_logos = []
trans = [transforms.ToPILImage(), transforms.Pad(10), 
    transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))]

##################################

for i, x in enumerate(files):
	image = PIL.Image.open(x)
	image = image.convert("RGB")
	image = np.asarray(image, dtype=np.float32) / 255
	image = image[:, :, :3]
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


#not_logo = "/home/antonio/Desktop/trill/images/pngs/ABN Newswire (Chinese - Simplified).png"

