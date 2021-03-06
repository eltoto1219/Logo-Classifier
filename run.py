from image_eval import *
from logo_nn import *
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as d
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

composed = transforms.Compose([transforms.Pad(10), 
    transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])
PATH = "/home/antonio/Desktop/trill/temp/"
TRILL_PATH = "/home/antonio/Desktop/trill/trill_logos"
data = datasets.ImageFolder(PATH, transform= composed)
trill_data = datasets.ImageFolder(TRILL_PATH, transform= composed)

random_seed = 20
num_data = len(data)
batch_size = 85
test_size  = .3
indices_data = list(range(num_data))

np.random.seed(random_seed)
np.random.shuffle(indices_data)
split = int(np.floor(test_size * num_data))
train_idx, test_idx = indices_data[split:], indices_data[:split]


train_sampler = d.SubsetRandomSampler(train_idx)
test_sampler = d.SubsetRandomSampler(test_idx)

train_loader = DataLoader(data, batch_size= batch_size, sampler=train_sampler, num_workers=2)
test_loader = DataLoader(data, sampler = test_sampler, batch_size=batch_size)

trill_loader = DataLoader(trill_data, num_workers = 2, batch_size = 85)


# if __name__ == "__main__":
#     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logoNet = Net().to(dev)
#     optimizer = optim.SGD(logoNet.parameters(), lr=0.001, momentum=0.001)
#     criterion = torch.nn.CrossEntropyLoss()

#     training(net = logoNet,
#     device = dev,
#     train_loader = train_loader,
#     optimizer = optimizer,
#     n_epoch= 1,
#     save_interval = 1,
#     loss_func = criterion)


