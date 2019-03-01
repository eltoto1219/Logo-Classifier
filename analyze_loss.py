import os
import numpy as np
from matplotlib import pyplot as plt
import torch 
import re
from run import *
from image_eval import *

class main:
	PATH = print(os.getcwd())

	#loading checkpoints from direcotry 

	CHECKPOINTS = "/home/antonio/Desktop/chkpnts"
	TEST_LOSS = "/home/antonio/output_loss.txt"

	#foo = torch.load(CHECKPOINTS, os.listdir(CHECKPOINTS)[0])

	foo = os.listdir(CHECKPOINTS)
	l = []

	for file in foo:
		file = str(file)
		temp = re.findall("\d+", file)
		temp = [temp[0], file]
		l.append(temp)

	test = sorted(l,key=lambda l: int(l[0]))

	loss = []
	epoch = []
		

	for _,y in test:
		epoch.append(int(_))
		TEMP = os.path.join(CHECKPOINTS, y)
		TEMP = torch.load(TEMP, map_location = "cpu")
		TEMP = TEMP['loss'].item()
		loss.append(TEMP)

	dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logoNet = Net().to(dev)
	optimizer = optim.SGD(logoNet.parameters(), lr=0.001, momentum=0.001)
	criterion = torch.nn.CrossEntropyLoss()

	# test_loss = []
	# for i, (_,y) in enumerate(test):
	# 	TEMP = os.path.join(CHECKPOINTS, y)
	# 	TEMP = torch.load(TEMP, map_location = "cpu")
	# 	TEMP = TEMP['model_state_dict']
	# 	logoNet.load_state_dict(TEMP)
	# 	loss_2 = testing(logoNet, dev, test_loader, criterion)
	# 	test_loss.append(np.mean(loss_2))

	
	print(len(epoch))
	print(len(loss))

	# with open(TEST_LOSS, 'w') as f:
	# 	for item in test_loss:
	# 		f.write("%s\n" % item)

	plt.plot(epoch, loss)
	#plt.plot(epoch, test_loss, label = "Test Loss")
	plt.title('Cross entropy loss of Softmax')
	#plt.legend(loc='upper left')
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.show()
	plt.savefig('/home/antonio/loss_figure.png')

quit()