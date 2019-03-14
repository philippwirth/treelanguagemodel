import torch
import torch.nn as nn


class RBFKernel(nn.Module):

	def __init__(self, x0, sigma, trainable=False):

		super(RBFKernel, self).__init__()
		self.x0 = x0
		self.sigma = sigma

	def forward(self, x):
		exponent = (x - self.x0).pow(2) / (2 * self.sigma * self.sigma)
		return torch.exp(-exponent)
