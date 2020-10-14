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

print(torch.cuda.is_available())
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))

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
			nn.Conv2d(1, 12, kernel_size=(3,3), stride=1, padding=(1,1)),
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


BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.0
EPS_END = 0.0
EPS_DECAY = 700000
TARGET_UPDATE = 30
NORMALISATION_CONST = 4000000
lr = 0.0005
manual_training = False
is_checkpoint = True
testing = True
checkpoint = None
if is_checkpoint:
	checkpoint = torch.load("ctrai_cv1.model")

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
steps_done_1 = 0
steps_done_2 = 0
steps_done_3 = 0
steps_done_4 = 0
steps_done_5 = 0
steps_done_6 = 0
steps_done_7 = 0
steps_done_8 = 0
previous_random_action = 0

def select_action_old(state):
	global steps_done
	global previous_random_action
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * max(0., (1. - (steps_done / EPS_DECAY)))
	steps_done += 1
	if sample > eps_threshold:
		with torch.no_grad():
			# t.max(1) will return largest column value of each row.
			# second column on max result is index of where max element was
			# found, so we pick action with the larger expected reward.
			return 0, policy_net(state).argmax(dim=1), eps_threshold
	else:
		if random.random() < 0.90: #increase likelihood of same random actions in a row
			action = previous_random_action
		else:
			action = (previous_random_action + random.randrange(1, n_actions)) % n_actions
		previous_random_action = action
		if manual_training:
			return 2, torch.tensor([action]), eps_threshold
		else:
			return 1, torch.tensor([action]), eps_threshold

def select_action(state, episode, theta):
	global steps_done
	global steps_done_1
	global steps_done_2
	global steps_done_3
	global steps_done_4
	global steps_done_5
	global steps_done_6
	global steps_done_7
	global steps_done_8
	global previous_random_action
	steps_done += 1
	if steps_done % 100000 == 0:
		print(steps_done)
	ratio = 0.
	if theta < -math.pi*3/4: #split track into 8 pie slices, each with its own decaying rate
		steps_done_1 += 1
		ratio = (steps_done_8 + steps_done_1 + steps_done_2)
	elif theta < -math.pi*2/4:
		steps_done_2 += 1
		ratio = (steps_done_1 + steps_done_2 + steps_done_3)
	elif theta < -math.pi*1/4:
		steps_done_3 += 1
		ratio = (steps_done_2 + steps_done_3 + steps_done_4)
	elif theta < 0:
		steps_done_4 += 1
		ratio = (steps_done_3 + steps_done_4 + steps_done_5)
	elif theta < math.pi*1/4:
		steps_done_5 += 1
		ratio = (steps_done_4 + steps_done_5 + steps_done_6)
	elif theta < math.pi*2/4:
		steps_done_6 += 1
		ratio = (steps_done_5 + steps_done_6 + steps_done_7) 
	elif theta < math.pi*3/4:
		steps_done_7 += 1
		ratio = (steps_done_6 + steps_done_7 + steps_done_8)
	else:
		steps_done_8 += 1
		ratio = (steps_done_7 + steps_done_8 + steps_done_1)
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * max(0., (1. - (ratio / EPS_DECAY)))
	if sample > eps_threshold:
		with torch.no_grad():
			return 0, policy_net(state).argmax(dim=1), eps_threshold
	else:
		if random.random() < 0.95: #increase likelihood of same random actions in a row
			action = previous_random_action
		else:
			action = (previous_random_action + random.randrange(1, n_actions)) % n_actions
		previous_random_action = action
		if manual_training:
			return 2, torch.tensor([action]), eps_threshold
		else:
			return 1, torch.tensor([action]), eps_threshold

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


print("test")
console = Console()
status = ""
first_connection = True
print("test")
image = TrackImage("C:/Users/Justin/Documents/CTR/BizHawk-2.4.1/crashcove_aiview.png", crop_size)
image.resize(10)

num_episodes = 10000

for i_episode in range(num_episodes):
	if first_connection:
		status = console.recv()
		if (status != "Connected"):
			print("ERROR: Couldn't connect to the lua script.")
			exit()
		first_connection = False

	inputs = list(map(float, console.recv().split()))
	#theta = np.arctan2(inputs[0], inputs[1])
	image.rotate_and_crop(inputs[0], inputs[1], inputs[2])

	state = T.ToTensor()(image.crop).unsqueeze_(0).to(device)

	#state = torch.tensor([inputs]).to(device)

	for t in count():
		is_random, action, random_threshold = select_action_old(state)
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

		inputs = list(map(float, console.recv().split()))
		#theta = np.arctan2(inputs[0], inputs[1])
		image.rotate_and_crop(inputs[0], inputs[1], inputs[2])

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
			}, "ctrai_cv1.model")

console.close()