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
	return torch.mul(dist, dist)		# square the distance and return it
	

def eucl_entailcone_dist(hidden, next_hidden):


	dist_fn = nn.PairwiseDistance(p=2)
	pw_dist = dist_fn(hidden, next_hidden)

	hidden = hidden.expand(next_hidden.size(0), -1)
	hidden_len = hidden.norm(p=2, dim=1)
	next_hidden_len = next_hidden.norm(p=2, dim=1)

	inpt1 = next_hidden_len.pow(2) - hidden_len.pow(2) - pw_dist.pow(2)
	inpt2 = torch.mul(torch.mul(hidden_len, 2), pw_dist)
	inpt2[inpt2 == 0] = 1e-5
	inpt = torch.div(inpt1, inpt2)

	print(inpt1)
	print(inpt2)
	print(inpt)
	print(torch.acos(inpt))

	return torch.acos(inpt)

