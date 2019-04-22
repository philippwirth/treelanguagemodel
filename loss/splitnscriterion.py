import torch
import torch.nn as nn

class SplitNegativeSampleCriterion(nn.Module):

	def __init__(self, temp=5):
		super(SplitNegativeSampleCriterion, self).__init__()
		self.temp = temp

	def forward(self, hiddens, outputs):
		'''
			compute maximilians loss for each pair of hidden state and output,
			index at 0 is positive sample
		'''

		# distance function is euclidean distance	
		dist_fn = nn.PairwiseDistance(p=2)

		loss = 0
		for hidden, output in zip(hiddens, outputs):
		
			# compute the distance
			dist = self.temp * dist_fn(hidden, output).pow(2)
			
			# positive sample
			pos = -dist[0]

			# negative samples
			dist = torch.exp(-dist)
			negs = torch.log(torch.sum(dist[1:]))

			loss = loss - (pos - negs)

		return loss / len(hiddens)


class NegativeSampleCriterion(nn.Module):

	def __init__(self, temp=5):
		super(NegativeSampleCriterion, self).__init__()
		self.temp = temp

	def forward(self, output):

		dist_fn = nn.PairwiseDistance(p=2)

		# seq_len includes the initial hidden states! look out for that
		seq_len, seq_len_times_nsamples, hsz = output.size()
		nsamples = (seq_len_times_nsamples-1) // (seq_len-1)	# minus 1 because initial hiddens

		# compute loss
		loss = 0
		for i in range(1, seq_len):
			# don't consider initial hidden states

			# get positive term
			pos = -self.temp * dist_fn(output[i-1][0], output[i][0].view(1,-1))

			# get negative term
			left, right = (i-1)*nsamples+1, i*nsamples+1
			dist = -self.temp * dist_fn(output[i-1][0], output[i][left:right])
			neg = torch.log(torch.sum(torch.exp(dist)))

			# update loss
			loss -= (pos - neg) / seq_len

		return loss


class SplitCrossEntropy(nn.Module):


	def __init__(self, ntokens, temp, splits):
		super(SplitCrossEntropy, self).__init__()
		
		self.ntokens = ntokens
		self.temp = temp
		
		self.splits = splits
		self.nsplits = len(splits) - 1

		self.tombstone_base = self.ntokens - self.nsplits + 1

		self.head_targets = torch.LongTensor([i for i in range(min(self.ntokens, self.splits[1]))])
		self.head_targets =self.head_targets.cuda()

		if self.nsplits > 1:
			self.tail_targets = torch.LongTensor([self.tombstone_base + i for i in range(self.nsplits-1)]) # minus 1 because head doesn't need repr.
			self.tail_targets = self.tail_targets.contiguous().cuda()

	def _log_probs(self, model, hidden, tokens, batch_size=128):

		dist_fn = nn.PairwiseDistance(p=2)

		# number of words we evaluate at once
		nbatch = len(tokens) // batch_size
		# if we can't divide evenly need one more batch
		nbatch = nbatch if (len(tokens) % batch_size) == 0 else nbatch + 1

		outputs = []
		for j in range(nbatch):

			# apply model to all words in the split
			ntokens = batch_size if (j+1)*batch_size <= len(tokens) else len(tokens) % batch_size
			print(str(j*batch_size + ntokens) + " and " + str(len(toekns)))
			hiddens = self._copy_hidden(hidden, ntokens)					# copy hidden state nbatch times
			token_batch = tokens[j*batch_size:j*batch_size + ntokens]		# get batch of words
			output, new_hidden = model(token_batch.view(1,-1), hiddens)	# evaluate
			outputs.append(output)

		# compute distances between input and outputs
		outputs = torch.cat(outputs, dim=0)
		dist = -self.temp * dist_fn(hidden, output).pow(2)

		softmaxed = torch.nn.functional.log_softmax(dist, dim=-1)

		return softmaxed, outputs

	def forward(self, model, target, hidden, batch_size=128):

		# target in head?
		target_in_head = target < self.splits[1]

		# bring hidden in correct form
		
		# head probs
		if self.nsplits > 1:
			head_targets = torch.cat((self.head_targets, self.tail_targets))
		else:
			head_targets = self.head_targets
		head_log_probs, outputs = self._log_probs(model, hidden, head_targets, batch_size=batch_size)

		if target_in_head:
			entropy = -head_log_probs[target]
			new_hidden = outputs[target]
		else:

			# find split
			i = 0
			while self.splits[i+1] <= target: i = i+1
			tombstone_idx = self.splits[1] + i - 1

			# get new hidden
			new_hidden = outputs[tombstone_idx]

			# compute tail log probabilities
			left, right = self.splits[i], min(self.splits[i+1], self.tombstone_base)
			tail_targets = torch.LongTensor([i for i in range(left, right)]).cuda()
			tail_log_probs, outputs = self._log_probs(model, new_hidden, tail_targets, batch_size=batch_size)

			# entropy
			head_entropy = head_log_probs[tombstone_idx]
			tail_entropy = tail_log_probs[target - self.splits[i]]
			entropy = -(head_entropy + tail_entropy)

			new_hidden = outputs[target - self.splits[i]]

		return entropy, new_hidden

	def _copy_hidden(self, hidden, n):

		# copy hidden s.t. nbatch is ntokens
		result = hidden.expand(n, -1)	# ntokens x hsz 
		result = result.view(1, n, -1)	# (n_layers*n_directions) x ntokens x hsz
		return [result.contiguous()]	# add another layer of brackets because this is expected input

