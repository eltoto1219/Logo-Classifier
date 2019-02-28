import os
import numpy as np
from matplotlib import pyplot as plt
import torch 
import re

class main:
	PATH = print(os.getcwd())

	#loading checkpoints from direcotry 

	CHECKPOINTS = "/home/antonio/Desktop/trill/project1/chkpnts"

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

	print(loss)

	plt.plot(epoch, loss)
	plt.title('Cross entropy loss of Softmax')
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.show()