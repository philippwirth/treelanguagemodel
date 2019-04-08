import torch
import torch.nn as nn

class SimpleEuclDistance(nn.Module):

	def __init__(self):
		
		super(SimpleEuclDistance, self).__init__()
		pass

	def forward(self, x, y):
		'''
			x of shape 1 x hsz
			y of shape n x hsz

			return: distance of x to every y_i, size: n
		'''

		dist_fn = nn.PairwiseDistance(p=2)
		return dist_fn(x, y)

class EuclideanDistance(nn.Module):

	def __init__(self):
		
		super(EuclideanDistance, self).__init__()
		pass

	def forward(self, hidden, next_hidden):
		''' Takes the last hidden state and the next hidden states (1 for each token) 
			and computes the distance between the hidden and each next hidden state.
			hidden: 		(bsz x hsz) the last hidden state
			next_hidden:	((bsz*ntokens) x hsz) next hidden state for each token

			return: distance matrix of the form (bsz x ntokens)
		'''

		# define distance function
		dist_fn = nn.PairwiseDistance(p=2)

		# find pairwise distances between hidden and next_hidden for the whole batch
		bsz = hidden.size(0)
		ntokens = next_hidden.size(0) // bsz
		distances = []
		for i in range(bsz):
			dist = dist_fn(hidden[i], next_hidden[i*ntokens:(i+1)*ntokens])
			distances.append(dist)

		# stack distances to matrix of size bsz x ntokens
		dist_m = torch.stack(distances)
		assert (dist_m.size(0) == bsz and dist_m.size(1) == ntokens)

		return dist_m
		


class EuclideanEntailmentConeDistance(nn.Module):

	def __init__(self):
		super(EuclideanEntailmentConeDistance, self).__init__()
		pass

	def forward(self, hidden, next_hidden):
		return 0


