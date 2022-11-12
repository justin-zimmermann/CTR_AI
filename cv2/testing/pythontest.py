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
from PIL import Image

class TrackImage():
	def __init__(self, path, crop_size):
		self.image = Image.open(path).convert("L")
		self.crop = None
		self.crop_size = crop_size

	def resize(self, scale):
		pass #TODO: resize the image to make it smaller and run the code faster
		self.image = self.image.reduce(scale)

	def get_x(self, x):
		return x*0.00024703 + 1096.3871668881495

	def get_y(self, y):
		return y*0.00024687 + 720.6058956589455

	def rotate_and_crop(self, a, b, angle):
		a = self.get_x(a)
		b = self.get_y(b)
		original_x, original_y = 2385., 1355.
		x, y = self.image.size
		a = a * x / original_x
		b = b * y / original_y
		theta = (-angle*360./4095.)
		self.crop = self.image.rotate(theta, expand=True)
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
		self.crop.save("C:/Users/Justin/Documents/CTR/CTR_AI/computer-vision/test.png")

scale = 1.
image = TrackImage("C:/Users/Justin/Documents/CTR/BizHawk-2.4.1/crashcove_aiview.png", 200./scale)
image.resize(scale)

image.rotate_and_crop(1794450, -1559565, 3118)

#image.image.save("C:/Users/Justin/Documents/CTR/CTR_AI/cv2/test.png")

image.save()

print(image.crop.getdata())

print(image.crop.size)  
 
pil_to_tensor = T.ToTensor()(image.crop).unsqueeze_(0)
print(pil_to_tensor.shape) 

tensor_to_pil = T.ToPILImage()(pil_to_tensor.squeeze_(0))
print(tensor_to_pil.size)

tensor_to_pil.save("C:/Users/Justin/Documents/CTR/CTR_AI/cv2/test2.png")





