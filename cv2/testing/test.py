import socket
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os
from PIL import Image

class TrackImage():
	def __init__(self, crop_size):
		self.tracks = ["crashcove",
		"rootubes",
		"tigertemple",
		"cocopark",
		"mysterycaves",
		"blizzardbluff",
		"sewerspeedway",
		"dingocanyon",
		"papupyramid",
		"dragonmines",
		"polarpass",
		"cortexcastle",
		"tinyarena",
		"has",
		"nginlabs",
		"oxidestation",
		"slidecoliseum",
		"turbotrack"]
		self.paths = []
		self.root = "C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens/"
		for track in self.tracks:
			self.paths.append([])
			for file in os.listdir("C:/Users/Justin/Documents/CTR/CTR_AI/cv2/CTR Screens"):
				if file.startswith(track + "_processed"):
					self.paths[-1].append(self.root + file)
		#simplified, needs to account for maps with several images
		self.images = [[Image.open(subpath).convert("RGB") for subpath in path] for path in self.paths]
		self.crop = None
		self.crop_size = crop_size
		self.coeff_x = [(1096.3871668881495, 0.00024703),
		(648.8907972725096, 0.00016692),
		(808.2214949463429, 0.00015567),
		(1067.4476422000566, 0.00022205),
		(553.2231225619722, 0.00010774),
		(679.2222767804917, 0.00016676),
		(462.3120376807265, 0.00012186),
		(249.2599706607075, 0.00020405),
		(714.8400101850771, 0.00019952),
		(612.209940534515, 0.00022046),
		(518.8995554270739, 0.00015246),
		(661.4728181950248, 0.00016822),
		(581.7717707204861, 0.00013887),
		(587.3032686666095, 0.00019695),
		(258.24418337102253, 0.00017157),
		(317.87574198039175, 0.0001749),
		(676.2005498317476, 0.00011718),
		(806.3893287263951, 0.00014062)
		]
		self.coeff_y = [(720.6058956589455, 0.00024687),
		(951.6204588111457, 0.00016692),
		(541.2663631660163, 0.00015532),
		(622.7371313353979, 0.00022273),
		(528.5968185620499, 0.00010798),
		(383.9597198513739, 0.00016832),
		(978.7461432835855, 0.0001207),
		(495.31945787523387, 0.00020356),
		(700.0342431302049, 0.00019857),
		(730.9846413033176, 0.00021962),
		(1128.8737994356275, 0.00015239),
		(560.0389412399034, 0.00016791),
		(491.5447563291968, 0.000139),
		(703.7373943101733, 0.0001968),
		(9.809855177047098, 0.00017184),
		(308.57698624296677, 0.00017524),
		(546.7318019831731, 0.00011678),
		(542.7879343976957, 0.00013991)
		]
		self.original_xy = [(2385., 1355.),
		(1238., 1274.),
		(1732., 1111.),
		(2098., 1270.),
		(1107., 1058.),
		(1677., 1249.),
		(924., 1072.),
		(1663., 1180.),
		(1292., 1260.),
		(1352., 1199.),
		(1272., 1291.),
		(1229., 1209.),
		(1050., 801.),
		(1485., 1181.),
		(1339., 1271.),
		(1287., 1226.),
		(1244., 974.),
		(1503., 1158.)
		]

	def resize(self, scale):
		self.images = [[subimage.reduce(scale) for subimage in image] for image in self.images]

	def get_x(self, x, track_id):
		#to update
		return x*self.coeff_x[track_id][1] + self.coeff_x[track_id][0]

	def get_y(self, y, track_id):
		#to update
		return y*self.coeff_y[track_id][1] + self.coeff_y[track_id][0]

	def rotate_and_crop(self, a, b, angle, progress, track_id):
		# get image based on progress and track id
		image_index = 0
		if track_id == 1:
			if (progress > 61000 or progress < 44000):
				image_index = 0
			else:
				image_index = 1
		if track_id == 8:
			if (progress < 59000):
				image_index = 0
			else:
				image_index = 1
		if track_id == 9:
			if (progress > 26000):
				image_index = 0
			else:
				image_index = 1
		if track_id == 11:
			if (progress < 85000 and progress > 60000):
				image_index = 0
			else:
				image_index = 1
		if track_id == 12:
			if (progress < 38000 or progress > 137000):
				image_index = 0
			else:
				image_index = 1
		if track_id == 13:
			if (progress < 63000 and progress > 43000):
				image_index = 2
			elif (progress > 62900 and progress < 100800) or (progress < 26000 and progress > 15000):
				image_index = 1	
			else:
				image_index = 0	
		if track_id == 14:
			if (progress < 89000 and progress > 69000):
				image_index = 0
			else:
				image_index = 1
		if track_id == 15:
			if (progress < 43000 or progress > 135000):
				image_index = 0
			elif progress > 95000:
				image_index = 1	
			elif progress > 66000:
				image_index = 2
			else:
				image_index = 3

		a = self.get_x(a, track_id)
		b = self.get_y(b, track_id)
		original_x, original_y = self.original_xy[track_id]
		x, y = self.images[track_id][image_index].size
		a = a * x / original_x
		b = b * y / original_y
		theta = (-angle*360./4095.)
		self.crop = self.images[track_id][image_index].rotate(theta, expand=True)
		new_x, new_y = self.crop.size
		r = math.sqrt(((a-x/2.)**2.) + (b-y/2.)**2.)
		alpha = 360.*math.atan((a-x/2.)/(b-y/2.))/(2*math.pi)
		if (b-y/2.) < 0:
			alpha += 180.
		#print((a-x/2.), (b-y/2.))
		a_p = new_x/2. + r*math.sin(2*math.pi*(theta + alpha)/360.)
		b_p = new_y/2. + r*math.cos(2*math.pi*(theta + alpha)/360.)
		#print(a, b, x, y, theta, r, alpha, new_x, new_y, a_p, b_p)
		self.crop = self.crop.crop((a_p-0.5*self.crop_size, b_p-0.1*self.crop_size, a_p+0.5*self.crop_size, b_p+0.9*self.crop_size))
		#self.crop = self.crop.crop((b_p-1000, a_p-1000, b_p+1000, a_p+1000))

	def save(self): #for testing purposes
		self.crop.save("C:/Users/Justin/Documents/CTR/CTR_AI/cv2/testing/test.png")

image = TrackImage(20)
image.resize(10)


image.rotate_and_crop(-177285, 3261, 1571, 51142, 2)

image.save()

print(image.crop.getdata())

print(image.crop.size)  
 
pil_to_tensor = T.ToTensor()(image.crop).unsqueeze_(0)
print(pil_to_tensor.shape) 

tensor_to_pil = T.ToPILImage()(pil_to_tensor.squeeze_(0))
print(tensor_to_pil.size)

tensor_to_pil.save("C:/Users/Justin/Documents/CTR/CTR_AI/cv2/testing/test2.png")





