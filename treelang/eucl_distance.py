import torch
import torch.nn as nn

class EuclideanDistance(nn.Module):

	def __init__(self):
		
		super(EuclideanDistance, self).__init__()
		pass

	def forward(self, hidden, next_hidden):
		''' Takes the last hidden state and the next hidden states (1 for each token) 
			and computes the distance between the hidden and each next hidden state.
			hidden: 		(1 x hsz) the last hidden state
			next_hidden:	(ntokens x hsz) next hidden state for each token

			return: squared distance between hidden and each next hidden (ntokens)
		'''

		dist_fn = nn.PairwiseDistance(p=2)	# define the distance function
		dist = dist_fn(hidden, next_hidden)	# compute distance of hidden to each next_hidden
		return dist
		


class EuclideanEntailmentConeDistance(nn.Module):

	def __init__(self):
		super(EuclideanEntailmentConeDistance, self).__init__()
		pass

	def forward(self, hidden, next_hidden):
		return 0


