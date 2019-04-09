from collections import defaultdict

import torch
import torch.nn as nn

import numpy as np
from treelang.eucl_kernel import SimplePolynomialKernel
from treelang.eucl_distance import SimpleEuclDistance

class SplitTCELoss(nn.Module):


	def __init__(self, ntokens, splits, bsz=10, temp=65, dist='sqrd'):

		super(SplitTCELoss, self).__init__()
		self.splits = [0] + splits + [100 * 1000000]
		self.nsplits = len(self.splits) - 1
		self.stats = defaultdict(list)
		self.ntokens = ntokens # this is the original number of tokens

		self.bsz = bsz
		self.temp = temp

		# distance functions
		self.distance = SimpleEuclDistance() if dist == 'sqrd' else None
		self.kernel = SimplePolynomialKernel()

		# Each of the splits that aren't in the head require a pretend token, we'll call them tombstones
		# The probability given to this tombstone is the probability of selecting an item from the represented split
		# We need one extra token for each split
		if self.nsplits > 1:
			self.tail_targets = torch.LongTensor([self.ntokens + i for i in range(self.nsplits-1)]) # minus 1 because head doesn't need repr.
			self.tail_targets = self.tail_targets.contiguous().cuda()


	def logprob(self, model, hiddens, words):

		logprobs = []
		nbatch = (len(words) // self.bsz) + 1

		for i in range(hiddens.size(0)):

			outputs = []
			for j in range(nbatch):

				# apply model to all words in the split
				nwords = self.bsz if j < nbatch - 1 else len(words) % self.bsz
				print('nwords: ' + str(nwords))
				hidden = self._copy_hidden(hiddens[i], nwords)			# copy hidden state nbatch times
				word_batch = words[j*self.bsz:j*self.bsz + nwords]		# get batch of words
				print(word_batch)
				output, hidden = model(word_batch.view(1,-1), hidden)	# evaluate
				outputs.append(output)

			# compute distances between input and outputs
			outputs = torch.cat(outputs, dim=0)
			d = self.distance(hiddens[i].view(1,-1), outputs)
			k = self.temp * self.kernel(d)

			softmaxed = torch.nn.functional.log_softmax(k, dim=-1)
			logprobs.append(softmaxed.view(1,-1))

		return torch.cat(logprobs, dim=0)
	
	def split_on_targets(self, hiddens, targets):
		# Split the targets into those in the head and in the tail
		split_targets = []
		split_hiddens = []

		# Determine to which split each element belongs (for each start split value, add 1 if equal or greater)
		# This method appears slower at least for WT-103 values for approx softmax
		#masks = [(targets >= self.splits[idx]).view(1, -1) for idx in range(1, self.nsplits)]
		#mask = torch.sum(torch.cat(masks, dim=0), dim=0)
		###
		# This is equally fast for smaller splits as method below but scales linearly
		mask = None
		for idx in range(1, self.nsplits):
			partial_mask = targets >= self.splits[idx]
			mask = mask + partial_mask if mask is not None else partial_mask
		###
		#masks = torch.stack([targets] * (self.nsplits - 1))
		#mask = torch.sum(masks >= self.split_starts, dim=0)
		for idx in range(self.nsplits):
		# If there are no splits, avoid costly masked select
			if self.nsplits == 1:
				split_targets, split_hiddens = [targets], [hiddens]
				continue
			# If all the words are covered by earlier targets, we have empties so later stages don't freak out
			if sum(len(t) for t in split_targets) == len(targets):
				split_targets.append([])
				split_hiddens.append([])
				continue
			# Are you in our split?
			tmp_mask = mask == idx
			split_targets.append(torch.masked_select(targets, tmp_mask))
			split_hiddens.append(hiddens.masked_select(tmp_mask.unsqueeze(1).expand_as(hiddens)).view(-1, hiddens.size(1)))
		return split_targets, split_hiddens


	def forward(self, model, hiddens, targets, verbose=False):

		model.eval()
		total_loss = None

		# first, split hiddens and targets
		split_targets, split_hiddens = self.split_on_targets(hiddens, targets)

		# then, softmax over head vocab and tombstones (only add tombstones if nsplits > 1)
		start, end = self.splits[0], min(self.splits[1], self.ntokens)
		head_words = torch.LongTensor([i for i in range(start, end)]).contiguous().cuda()

		if self.nsplits > 1:
			head_words = torch.cat((head_words, self.tail_targets))
		
		# Perform the softmax calculation for the word vectors in the head for all splits
		# We need to guard against empty splits as torch.cat does not like random lists
		combo = torch.cat([split_hiddens[i] for i in range(self.nsplits) if len(split_hiddens[i])])
		###
		softmaxed_all_head_res = self.logprob(model, combo, head_words)
		running_offset = 0
		for idx in range(self.nsplits):
			# If there are no targets for this split, continue
			if len(split_targets[idx]) == 0: continue

			# For those targets in the head (idx == 0) we only need to return their loss
			if idx == 0:
				softmaxed_head_res = softmaxed_all_head_res[running_offset:running_offset + len(split_hiddens[idx])]
				entropy = -torch.gather(softmaxed_head_res, dim=1, index=split_targets[idx].view(-1, 1))
			# If the target is in one of the splits, the probability is the p(tombstone) * p(word within tombstone)
			else:
				softmaxed_head_res = softmaxed_all_head_res[running_offset:running_offset + len(split_hiddens[idx])]

				# Calculate the softmax for the words in the tombstone
				start, end = self.splits[idx], min(self.splits[idx + 1], self.ntokens)
				print('start, end = ' + str(start) + ', ' + str(end))
				words = torch.LongTensor([i for i in range(start, end)]).contiguous().cuda()
				print('len(words) = ' + str(len(words)))
				softmaxed_tail_res = self.logprob(model, split_hiddens[idx], words)

				# Then we calculate p(tombstone) * p(word in tombstone)
				# Adding is equivalent to multiplication in log space
				head_entropy = softmaxed_head_res[:, -idx]
				# All indices are shifted - if the first split handles [0,...,499] then the 500th in the second split will be 0 indexed
				indices = (split_targets[idx] - self.splits[idx]).view(-1, 1)
				# Warning: if you don't squeeze, you get an N x 1 return, which acts oddly with broadcasting
				tail_entropy = torch.gather(softmaxed_tail_res, dim=1, index=indices).squeeze()
				entropy = -(head_entropy + tail_entropy)
			###
			running_offset += len(split_hiddens[idx])
			total_loss = entropy.float().sum() if total_loss is None else total_loss + entropy.float().sum()

		model.train()
		return (total_loss / len(targets)).type_as(model.decoder.weight)


	def _copy_hidden(self, hidden, n):

		# copy hidden s.t. nbatch is ntokens
		result = hidden.repeat(n, 1)	# ntokens x hsz 
		result = result.view(1, n, -1)	# (n_layers*n_directions) x ntokens x hsz
		return [result]								# add another layer of brackets because this is expected input
