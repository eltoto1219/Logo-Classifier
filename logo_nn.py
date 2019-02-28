import torch 
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 15, 5)
        #we input 6 channel. We want 15 channel output (so 6 different filters) 5 X 5 filter, stride 1, 0 padding
        self.conv2 = nn.Conv2d(15, 30, 5)
        #done with convolutions, now fully connected layer time. We want 16 (num channels) * 5^2 (area of filter) flattened IN, we want 120 OUT
        self.fc1 = nn.Linear(25230, 256)
        #120 in, 84 out
        self.fc2 = nn.Linear(256, 84)
        #84 in, 84 out
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.Softmax(dim = 1)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, of x dim you can only specify a single number
        #feeds into conv2. image size doesnt matter because all conv 2 is looking for is the number of filters, (depth)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 
        #now we connect to fully connected layer so we need to flatten the dimensions of all the filters in the last layer
        x = x.view(-1, self.num_flat_features(x))
        #now we pplying acitvation to fully connected layers.
        #print("Num flat features is", x.shape)
        x = F.relu(self.fc1(x))
        #print("The size of FC 1 is", x.shape)
        x = F.relu(self.fc2(x))
        #print("The size of FC 2 is", x.shape)
        x = self.fc3(x)
       # print("The size of FC 3 is", x.shape)
        x = self.softmax(x)
        #print("The softmax shape should be the same", x.shape)
        return x

    def num_flat_features(self, x):
        size = x.shape[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
