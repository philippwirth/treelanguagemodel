import torch
import torch.nn as nn

class DotProduct(nn.Module):

	def __init__(self):
		super(DotProduct, self).__init__()

	def forward(self, hidden, next_hidden):
		''' Takes the last hidden state and the next hidden states (1 for each token) 
			and computes the distance dotproduct between them.
			hidden: 		(bsz x hsz) the last hidden state
			next_hidden:	((bsz*ntokens) x hsz) next hidden state for each token

			return: dot product matrix of the form (bsz x ntokens)
		'''

		bsz = hidden.size(0)
		ntokens = next_hidden.size(0) // bsz
		dot_products = []
		for i in range(bsz):
			dots = torch.mv(next_hidden[i*ntokens:(i+1)*ntokens],hidden[i])
			dot_products.append(dots)

		# stack distances to matrix of size bsz x ntokens
		dot_product_matrix = torch.stack(dot_products)
		assert (dot_product_matrix.size(0) == bsz and dot_product_matrix.size(1) == ntokens)

		return dot_product_matrix
		

class PolynomialKernel(nn.Module):

	def __init__(self, x0=0.0, p=2):

		super(PolynomialKernel, self).__init__()
		self.p, self.x0 = p, x0

	def forward(self, x):
		return -(x - self.x0).pow(2)

class RBFKernel(nn.Module):

	def __init__(self, x0, sigma, trainable=False):

		super(RBFKernel, self).__init__()
		self.x0 = x0
		self.sigma = sigma

	def forward(self, x):
		exponent = (x - self.x0).pow(2) / (2 * self.sigma * self.sigma)
		return torch.exp(-exponent)
