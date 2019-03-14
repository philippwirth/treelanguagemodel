import torch
import torch.nn as nn


def eucl_dist_square(hidden, next_hidden):
	''' Takes the last hidden state and the next hidden states (1 for each token) 
		and computes the distance between the hidden and each next hidden state.
		hidden: 		(1 x hsz) the last hidden state
		next_hidden:	(ntokens x hsz) next hidden state for each token

		return: squared distance between hidden and each next hidden (ntokens)
	'''

	dist_fn = nn.PairwiseDistance(p=2)	# define the distance function
	dist = dist_fn(hidden, next_hidden)	# compute distance of hidden to each next_hidden
	return torch.mul(0.1 - dist, 0.1 - dist)		# square the distance and return it
	

def eucl_entailcone_dist(hidden, next_hidden):


	return 0

