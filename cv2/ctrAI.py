import socket
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

print(torch.cuda.is_available())
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))

#==================================================================
#                    CLASS DEFINITIONS
#==================================================================

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
		(1339., 1400.),
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
			if (progress < 53000):
				image_index = 0
			elif (progress < 59000):
				image_index = 2
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
			elif progress > 83000:
				image_index = 2
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
		if track_id == 16:
			if (progress < 84000 and progress > 40000):
				image_index = 2
			elif (progress < 40001 and progress > 21000):
				image_index = 1
			else:
				image_index = 0
		if track_id == 17:
			if (progress < 41000 and progress > 36500):
				image_index = 1
			elif (progress < 36501 and progress > 32000):
				image_index = 0
			else:
				image_index = 2

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
		self.crop.save("C:/Users/Justin/Documents/CTR/CTR_AI/cv2/test.png")




class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):

	def __init__(self, crop_size, outputs):
		super(DQN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 12, kernel_size=(3,3), stride=1, padding=(1,1)),
			nn.ReLU())
		self.layer2 = nn.Sequential(
			nn.Conv2d(12, 16, kernel_size=(5,5), stride=1, padding=(2,2)),
			nn.ReLU())
		self.layer3 = nn.Sequential(
			nn.Conv2d(16, 12, kernel_size=(3,3), stride=1, padding=(1,1)),
			nn.ReLU())
		self.layer4 = nn.Sequential(
			nn.Conv2d(12, 1, kernel_size=(1,1), stride=1, padding=(0,0)),
			)
		self.fc1 = nn.Linear(crop_size*crop_size, 32)
		self.fc2 = nn.Linear(32, outputs)
		self.crop_size = crop_size

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(-1, self.crop_size*self.crop_size)
		out = F.relu(self.fc1(out))
		out = self.fc2(out)
		return out

class Console:
	def __init__(self, host = '127.0.0.1', port = 8001):
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sock.bind((host, port))
		self.sock.listen(1)
		self.connection, self.address = self.sock.accept()
		print('Python got a client at {}'.format(self.address))

	def recv(self):
		self.buffer = self.connection.recv(1024).decode()
		return self.buffer

	def send(self, msg):
		_ = self.connection.send(msg.encode())

	def close(self):
		_ = self.connection.close()

#========================================================================
#            TURNING MODEL PARAMETERS
#========================================================================

BATCH_SIZE = 64
GAMMA = 0.999
RANDOMNESS = 0.0
MIN_RANDOMNESS = 0.0
MAX_RANDOMNESS = 1.0
TARGET_UPDATE = 30
lr = 0.0005
manual_training = False
is_checkpoint = True
testing = True
checkpoint = None
if is_checkpoint:
	checkpoint = torch.load("models/ctrai_cv2_450_2_92.model")

crop_size = 20
n_actions = 3

policy_net = DQN(crop_size, n_actions).to(device)
if is_checkpoint:
	policy_net.load_state_dict(checkpoint['model_state_dict'])
	#for name, param in policy_net.named_parameters():
		#if param.requires_grad:
			#print(name, param.data)
target_net = DQN(crop_size, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


optimizer = optim.RMSprop(policy_net.parameters(), lr=lr)
if is_checkpoint:
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	memory = checkpoint['memory']
else:
	memory = ReplayMemory(1000000)

steps_done = 0
previous_random_action = 0


#=========================================================
#               MORE FUNCTIONS
#=========================================================

def select_action(state):
	global steps_done
	global previous_random_action
	steps_done += 1
	if steps_done % 50000 == 0:
		print(steps_done)
	
	sample = random.random()
	if sample > RANDOMNESS:
		with torch.no_grad():
			#print(policy_net(state))
			return 0, policy_net(state).argmax(dim=1), RANDOMNESS
	else:
		if random.random() < 0.90: #increase likelihood of same random actions in a row
			action = previous_random_action
		else:
			action = (previous_random_action + random.randrange(1, n_actions)) % n_actions
		previous_random_action = action
		if manual_training:
			return 2, torch.tensor([action]), RANDOMNESS
		else:
			return 1, torch.tensor([action]), RANDOMNESS

def optimize_model():
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))

	next_state_batch = torch.cat(batch.next_state)
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	
	state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(-1))


	next_state_values = target_net(next_state_batch).max(1)[0].detach()
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1) #is that really good?
	optimizer.step()


#=================================================================
#                      MAIN LOOP
#=================================================================

print("Starting connection to Lua...")
console = Console()
status = ""
first_connection = True
print("Connection established. Please run the Lua script.")
image = TrackImage(crop_size)
image.resize(6)

num_episodes = 10000
print("test: ctrai_cv2_%s.model" % str(num_episodes))

for i_episode in range(num_episodes):
	if first_connection:
		status = console.recv()
		if (status != "Connected"):
			print("ERROR: Couldn't connect to the lua script.")
			exit()
		first_connection = False
	inputs = list(map(float, console.recv().split()))
	print(inputs)
	#theta = np.arctan2(inputs[0], inputs[1])
	image.rotate_and_crop(inputs[0], inputs[1], inputs[2], inputs[3], int(inputs[4]))

	state = T.ToTensor()(image.crop).unsqueeze_(0).to(device)

	#state = torch.tensor([inputs]).to(device)

	for t in count():
		is_random, action, random_threshold = select_action(state)
		#if t % 100 == 0:
		#	print(str(action.item()), inputs)

		console.send(str(action.item())+"\n")
		console.send(str(is_random) +"\n")
		console.send(str(random_threshold) +"\n")
		if is_random == 2:
			action = torch.tensor([int(console.recv())])

		reward_float = float(console.recv())
		reward = torch.tensor([reward_float], device=device)
		console.send("Reward received\n")
		if (reward_float == 5000):
			break
		if reward_float <= -50.:
			RANDOMNESS = min(MAX_RANDOMNESS, RANDOMNESS + 0.50)
		elif reward_float < 0.: #increase randomness if AI gets it wrong
			RANDOMNESS = min(MAX_RANDOMNESS, RANDOMNESS + 0.03)
		else: #quickly put back randomness to zero if AI gets it right
			RANDOMNESS = max(MIN_RANDOMNESS, RANDOMNESS - 0.15)

		inputs = list(map(float, console.recv().split()))
		#theta = np.arctan2(inputs[0], inputs[1])
		image.rotate_and_crop(inputs[0], inputs[1], inputs[2], inputs[3], int(inputs[4]))

		next_state = T.ToTensor()(image.crop).unsqueeze_(0).to(device)
		#next_state = torch.tensor([inputs]).to(device)

		if testing == False:
			memory.push(state, action, next_state, reward)

		state = next_state

		if testing == False:
			optimize_model()

	if (i_episode % TARGET_UPDATE == 0) and (testing == False):
		target_net.load_state_dict(policy_net.state_dict())
		torch.save({
			'model_state_dict': policy_net.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'memory': memory
			}, "ctrai_cv2_%s.model" % str(i_episode))

console.close()