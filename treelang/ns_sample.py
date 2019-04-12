import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

class SimpleSampler(nn.Module):

	def __init__(self, nsamples, frequencies, replacement=True):

		self.nsamples = nsamples
		self.frequencies = frequencies
		self.replacement = replacement
		super(SimpleSampler, self).__init__()

	def forward(self, args):

		# sample indices
		wrs = WeightedRandomSampler(self.frequencies, self.nsamples, replacement=self.replacement)

		# transform to tensor
		negs = torch.LongTensor(list(wrs)).cuda() if args.cuda else torch.LongTensor(list(wrs))

		return negs

		



