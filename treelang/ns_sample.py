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

class SplitNegativeSampler(nn.Module):

	def __init__(self, nsamples, frequencies, splits, exp=0.75):

		self.nsamples = nsamples
		self.frequencies = (frequencies / torch.sum(frequencies)).pow(exp)
		self.ntokens = len(self.frequencies)

		self.splits = splits
		self.nsplits = len(self.splits) - 1
		self.tombstones = torch.LongTensor([i for i in range(self.ntokens - self.nsplits + 1, self.ntokens)])

		super(SplitNegativeSampler, self).__init__()

	def forward(self, split, cuda):

		# mask out samples from other splits
		mask = torch.zeros(self.frequencies.size())
		index = torch.LongTensor([i for i in range(self.splits[split], self.splits[split+1])])
		if split == 0:
			# need tombstones in the split
			index = torch.cat((index, self.tombstones))
		mask.scatter_(0, index, 1.)

		# use masked frequencies
		masked_freqs = mask * self.frequencies
		print(masked_freqs)

		# get a sampler and sample negatives
		wrs = WeightedRandomSampler(masked_freqs, self.nsamples)
		negs = torch.LongTensor(list(wrs))
		print(split)
		print([self.splits[split], self.splits[split+1]])
		print(negs)

		return negs.cuda() if cuda else negs



		



