import torch
import torch.nn as nn

class NSLoss(nn.Module):

	def __init__(self, temp=5):
		super(NSLoss, self).__init__()
		self.temp = temp

	def forward(self, hidden, output):
		'''
			compute maximilians loss, index at 0 is positive sample
		'''

		# compute squared distances
		dist_fn = nn.PairwiseDistance(p=2)
		dist = self.temp*dist_fn(hidden, output).pow(2)

		# store positive sample
		pos = -dist[0]

		# sum and take log of negative samples
		dist = torch.exp(-dist)
		negs = torch.log(torch.sum(dist[1:]))

		# compute maximilians loss
		return -(pos - negs)

class SimpleEvaluationLoss(nn.Module):


	def __init__(self, ntokens):
		super(SimpleEvaluationLoss, self).__init__()
		self.ntokens = ntokens

	def forward(self, model, target, hidden):

		# initialize words and hidden states
		words = torch.LongTensor([i for i in range(self.ntokens)]).cuda()
		hiddens = hidden.repeat(self.ntokens, 1)

		# apply model to all words
		output, new_hidden = model(words.view(1,-1), [hiddens.view(1, self.ntokens, -1)])

		# compute distances
		dist_fn = nn.PairwiseDistance(p=2)
		dist =  -self.temp * dist_fn(hidden, output).pow(2)

		# compute crossentropy
		loss = nn.CrossEntropyLoss()
		return loss(dist.view(1,-1), target.view(1)), output, new_hidden


