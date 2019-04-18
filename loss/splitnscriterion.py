import torch
import torch.nn as nn

class SplitNegativeSampleCriterion(nn.Module):

	def __init__(self, temp=5):
		super(SplitNegativeSampleCriterion, self).__init__()
		self.temp = temp

	def forward(self, hiddens, outputs):
		'''
			compute maximilians loss for each pair of hidden state and output,
			index at 0 is positive sample
		'''

		# distance function is euclidean distance	
		dist_fn = nn.PairwiseDistance(p=2)

		loss = 0
		for hidden, output in zip(hiddens, outputs):
		
			# compute the distance
			dist = self.temp * dist_fn(hidden, output).pow(2)
			
			# positive sample
			pos = -dist[0]

			# negative samples
			dist = torch.exp(-dist)
			negs = torch.log(torch.sum(dist[1:]))

			loss = loss - (pos - negs)

		return loss / len(hiddens)

class SimpleEvaluationLoss(nn.Module):


	def __init__(self, ntokens, temp):
		super(SimpleEvaluationLoss, self).__init__()
		self.ntokens, self.temp = ntokens, temp

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
		return loss(dist.view(1,-1), target.view(1)), output


