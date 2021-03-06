import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


from treelang.eucl_distance import EuclideanDistance
from treelang.eucl_kernel import RBFKernel, PolynomialKernel, DotProduct

class TreelangCrossEntropyLoss(nn.Module):
	''' Computes cross entropy based on the treelang model: p(w|c) ~ e^-d(c, [w,c])^2
	''' 

	def __init__(self, ntokens=3, temp=100, distance='eucl', kernel='polynomial2'):

		super(TreelangCrossEntropyLoss, self).__init__()

		self.ntokens = ntokens
		self.words = torch.LongTensor([i for i in range(self.ntokens)])
		
		if distance == 'eucl':
			self.distance = EuclideanDistance()
		else:
			pass

		if kernel == 'polynomial1':
			self.kernel = PolynomialKernel(x0=0, p=1)
		if kernel == 'polynomial2':
			self.kernel = PolynomialKernel(x0=0, p=2)
		elif kernel == 'dot':
			self.kernel = DotProduct()
		else:
			pass
			
		self.temp = temp
		self.loss = nn.CrossEntropyLoss()
		self.kernel_type = kernel

	def forward(self, model, hiddens, targets, verbose=False):
		'''
			model: RNN
			hiddens: outputs of RNN for t in 1...T-1 ()
			targets: words of seq at time t in 2...T ()
		'''

		# find bsz (number of sequences done in parallel)
		bsz = 1 if len(targets.size()) < 2 else targets.size(1)

		# find sequence length
		seq_len = targets.size(0)
		
		# words to cuda (words becomes a 1 x (ntokens * bsz) vector)
		words = self.words.expand(bsz, -1).contiguous()
		words = words.view(1, self.ntokens * bsz)
		words = words.cuda()

		# for i in range seq_len! do all this
		total_loss = 0
		for i in range(seq_len):

			#last_hidden has size (bsz x hsz) -> bring it to 1 x (ntokens * bsz) x hsz
			last_hidden = hiddens[i]
			h = last_hidden.repeat(self.ntokens, 1)	# (ntokens * bsz) x hsz but wrong order
			index = torch.LongTensor(np.concatenate([bsz * np.arange(self.ntokens) + i for i in range(bsz)]))
			h = torch.index_select(h, 0, index.cuda()) # reorder
			h = [h.contiguous().view(1, self.ntokens * bsz, -1)]

			# forward pass through RNN to get output (1*bsz*n_words, ndir*hsz)
			output, hidden = model(words, h)

			if self.kernel_type in ['polynomial1', 'polynomial2']:
				d = self.distance(last_hidden, output)
				k = self.temp * self.kernel(d)
			else:
				k = self.temp*self.kernel(last_hidden, output)

			# use CrossEntropyLoss to compute the loss and average
			# input is of size (bsz x n_words)
			if bsz == 1:
				total_loss += self.loss(k.view(1, self.ntokens), targets[i].view(1))
			else:
				total_loss += self.loss(k, targets[:][i])

		return (total_loss / seq_len).type_as(model.decoder.weight)


if __name__ == '__main__':


	criterion = nn.CrossEntropyLoss()
	logsoftmax = nn.LogSoftmax()
	softmax = nn.Softmax()

	d = -1000*torch.tensor([0.01, 0.02])
	print(softmax(d))
	print(criterion(d.view(1,2), torch.LongTensor([1]).view(1)))


	'''
	target = torch.empty(1, dtype=torch.long).random_(4)
	pdist = nn.PairwiseDistance(p=2)
	input1 = torch.randn(1, 2)
	input2 = torch.randn(4, 2)


	input1.requires_grad_(True)
	input2.requires_grad_(True)

	optimizer = torch.optim.SGD([input1, input2], lr=0.1)
	losses = []
	for i in range(500):

		optimizer.zero_grad()

		output = pdist(input1, input2)
		output = -torch.mul(output, output)
		
		loss = criterion(output.view(1,4), target)
		print(loss)
		losses.append(loss.data)

		loss.backward()

		plt.plot(input2.t().detach().numpy()[0,:], input2.t().detach().numpy()[1,:],  marker='o', linestyle='None', color=plt.cm.viridis(i / 500))
		plt.plot(input1.detach().numpy()[0][0], input1.detach().numpy()[0][1], marker='o', linestyle='None', color=plt.cm.plasma(i / 500))
		

		optimizer.step()
	print(input1)
	print(input2)
	print(target)
	plt.show()

	'''
