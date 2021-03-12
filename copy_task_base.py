from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from MLP import MLPLayer
import argparse
from RNN import RNNLayer
from LSTM import LSTMLayer

print(np.__version__)
print(torch.__version__)

# Set the seed of PRNG manually for reproducibility
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)


# Copy data
def copy_data(T, K, batch_size):
	seq = np.random.randint(1, high=9, size=(batch_size, K))
	zeros1 = np.zeros((batch_size, T))
	zeros2 = np.zeros((batch_size, K - 1))
	zeros3 = np.zeros((batch_size, K + T))
	marker = 9 * np.ones((batch_size, 1))

	x = torch.LongTensor(np.concatenate((seq, zeros1, marker, zeros2), axis=1))
	y = torch.LongTensor(np.concatenate((zeros3, seq), axis=1))

	return x, y


# one hot encoding
def onehot(out, input):
	out.zero_()
	in_unsq = torch.unsqueeze(input, 2)
	out.scatter_(2, in_unsq, 1)


# Class for handling copy data
class RNNModel(nn.Module):
	def __init__(self, m, k):
		super(RNNModel, self).__init__()
		self.m = m
		self.k = k
		self.name = 'RNN'
		self.rnn = nn.RNNCell(m + 1, k)
		self.V = nn.Linear(k, m)
		# loss for the copy data
		self.loss_func = nn.CrossEntropyLoss()

	def forward(self, inputs):
		state = torch.zeros(inputs.size(0), self.k, requires_grad=False)

		outputs = []

		for input in torch.unbind(inputs, dim=1):
			state = self.rnn(input, state)
			outputs.append(self.V(state))

		return torch.stack(outputs, dim=1)

	def loss(self, logits, y):
		return self.loss_func(logits.view(-1, 9), y.view(-1))

	def accuracy_check(self, logits, y, K):
		yt = torch.argmax(logits, dim=2)
		correct = yt[:, -K:] == y[:, -K:]
		return float(correct.sum().item()) / (batch_size * K)


# Class for handling copy data
class LSTMModel(nn.Module):
	def __init__(self, m, k):
		super(LSTMModel, self).__init__()
		self.m = m
		self.k = k
		self.name = 'LSTM'
		self.lstm = nn.LSTMCell(m + 1, k)
		self.V = nn.Linear(k, m)
		# loss for the copy data
		self.loss_func = nn.CrossEntropyLoss()

	def forward(self, inputs):
		state = torch.zeros(inputs.size(0), self.k, requires_grad=False)
		state_1 = torch.zeros(inputs.size(0), self.k, requires_grad=False)

		outputs = []

		for input in torch.unbind(inputs, dim=1):
			state, c_t = self.lstm(input, (state, state_1))
			outputs.append(self.V(state))

		return torch.stack(outputs, dim=1)

	def loss(self, logits, y):
		return self.loss_func(logits.view(-1, 9), y.view(-1))

	def accuracy_check(self, logits, y, K):
		yt = torch.argmax(logits, dim=2)
		correct = yt[:, -K:] == y[:, -K:]
		return float(correct.sum().item()) / (batch_size * K)


# Class for handling copy data
class MLPModel(nn.Module):
	def __init__(self, c, k):
		super(MLPModel, self).__init__()
		self.c = c
		self.name = 'MLP'
		self.mlp = MLPLayer((T + 2 * K) * (c + 1), (T + 2 * K) * c)
		# loss for the copy data
		self.loss_func = nn.CrossEntropyLoss()

	def forward(self, inputs):
		input = inputs.reshape(batch_size, -1)
		return self.mlp(input)

	def loss(self, logits, y):
		logits_x = logits.reshape(batch_size, T + 2 * K, n_classes)
		return self.loss_func(logits_x.view(-1, 9), y.view(-1))

	def accuracy_check(self, logits, y, K):
		logits_x = logits.reshape(batch_size, T + 2 * K, n_classes)
		yt = torch.argmax(logits_x, dim=2)
		correct = yt[:, -K:] == y[:, -K:]
		return float(correct.sum().item()) / (batch_size * K)


T = 20  # Number of zeros after the random numbers
K = 3  # Number of number at the beginning
# K-1 : number of zeros after the delimiter ':'

batch_size = 32
iter = 5000
n_train = iter * batch_size
n_classes = 9
hidden_size = 64
n_characters = n_classes + 1
lr = 1e-3
print_every = 20

model_name = ""

colors = {'MLP': 'red', 'RNN': 'blue', 'LSTM': 'purple'}

models_ = {'MLP': MLPModel, 'RNN': RNNModel, 'LSTM': LSTMModel}

def main():
	models = [models_[model_name]]
	Accuracy = []

	for model in models:
		# create the training data
		X, Y = copy_data(T, K, n_train)
		print('{}, {}'.format(X.shape, Y.shape))

		ohX = torch.FloatTensor(batch_size, T + 2 * K, n_characters)
		onehot(ohX, X[:batch_size])
		print('{}, {}'.format(X[:batch_size].shape, ohX.shape))

		model = model(n_classes, hidden_size)
		model.train()

		opt = torch.optim.RMSprop(model.parameters(), lr=lr)

		loss_values = np.zeros(iter)
		baseline = (10 * np.log(8)) / (T + 2 * K)

		for step in range(iter):
			bX = X[step * batch_size: (step + 1) * batch_size]
			bY = Y[step * batch_size: (step + 1) * batch_size]

			onehot(ohX, bX)
			opt.zero_grad()
			logits = model(ohX)
			loss = model.loss(logits, bY)
			loss.backward()
			opt.step()

			loss_values[step] = loss.item()

			if step % print_every == 0:
				acc = model.accuracy_check(logits, bY, K)
				print('Step={}, Loss={:.4f} Accuracy={:.4f}'.format(step, loss.item(), acc * 100))

		plt.plot(loss_values, colors[model.name], label=model.name)
		Accuracy.append([model.name, acc])
		print(Accuracy)

		# Save
		torch.save(model, model_name + "_model.pt")

	plt.title("Time Lag {}".format(T))
	plt.axhline(y=float(baseline), label='baseline')
	plt.xlabel('Training Examples')
	plt.ylabel('Cross Entropy')

	plt.legend()
	plt.show()


def get_command_line_args():
	"""
	Returns the application arguments parser.
	"""
	parser: ArgumentParser = argparse.ArgumentParser()
	parser.add_argument('-t', '--t_blank', help="Number of blanks until delimiter", default=20)
	parser.add_argument('-k', '--k_copynum', help="Number of numbers to copy", default=3)
	parser.add_argument('-m', '--model', help="Choose model RNN, LSTM or MLP", default='MLP')
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = get_command_line_args()
	T = int(args.t_blank)
	K = int(args.k_copynum)
	model_name = args.model
	print(f"T: {T}, K: {K}, model_name: {model_name}")
	main()
