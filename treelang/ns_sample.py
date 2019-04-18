import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

class SimpleSampler(nn.Module):

	def __init__(self, nsamples, frequencies, replacement=True):

		self.nsamples = nsamples
		self.frequencies = frequencies.pow(0.5)# / torch.sum(frequencies)).pow(0.5)
		self.replacement = replacement
		super(SimpleSampler, self).__init__()

	def forward(self, token, args, allow_self=True):

		if not allow_self:
			self_freq = self.frequencies[token]
			self.frequencies[token] = 0

		# sample indices
		wrs = WeightedRandomSampler(self.frequencies, self.nsamples, replacement=self.replacement)

		if not allow_self:
			self.frequencies[token] = self_freq

		# transform to tensor
		negs = torch.LongTensor(list(wrs)).cuda() if args.cuda else torch.LongTensor(list(wrs))

		return negs

		



