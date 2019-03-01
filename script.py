import numpy as np
import random

PATH = "/home/antonio/Downloads/b_lovely_landscapes.txt"
OUTPUT = "/home/antonio/Downloads/outpt.txt"

with open(PATH, "r") as file:
    data = file.read().split("\n")
data_n = []
verts = []
num = data[0]
data.remove(num)
for idx, line in enumerate(data):
    if line.strip() == "":
        continue
    vals = [x.strip() for x in line.split(" ")]
    slide = {"id": idx, "data": set(vals[2:])}
    if (vals[0] == "H"):
        data_n.append(slide)
    else:
        verts.append(slide)

result = []
slide = np.random.choice(data_n)
count = 0

while(len(data_n)):
	data_n.remove(slide)
	result.append(slide)
	common = 0
	while(count < 10000000 | common == 0):
		slide_2 = np.random.choice(data_n)
		common = set(slide['data']).intersection(slide_2['data'])
		common = len(common)
	count += 1
	if 'slide_2' in locals():
		slide = slide_2
	else:
		break

# num_slides = len(result)

print(result)

# with open(OUTPUT, "w") as out:
#     out.write(str(num_slides) + "\n")
#     for val in result:
#         out.write(str(val["id"]) + "\n")

# print("Done")