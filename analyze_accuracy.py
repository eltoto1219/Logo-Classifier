import torch 
from image_eval import *
from logo_nn import *
from run import *
import os



#setting up dev, nn, optimizer, and loss function
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logoNet = Net().to(dev)
optimizer = optim.SGD(logoNet.parameters(), lr=0.001, momentum=0.001)
criterion = torch.nn.CrossEntropyLoss()

#testing(logoNet, dev, test_loader, criterion)

#accuracy after loaded weights

checkpoint = os.path.join(os.getcwd(), "chkpnts/model_chkpnt_epoch_133_.tar")
checkpoint = torch.load(checkpoint, map_location = "cpu")
logoNet.load_state_dict(checkpoint['model_state_dict'])


testing(logoNet, dev, test_loader, criterion)




