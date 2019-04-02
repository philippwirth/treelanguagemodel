import time
import math
import numpy as np
import torch
import torch.nn as nn
import random

# super
from language_models.abstract_language_model import AbstractLanguageModel

# loss and model
from treelang.crossentropy import TreelangCrossEntropyLoss
from merity.model import RNNModel

# utils
from merity.utils import batchify, get_batch, repackage_hidden
import merity.data as data
from visualize.dump import dump_contexts


'''
	Abstract class for all tree language models.
'''
class AbstractTreeLanguageModel(AbstractLanguageModel):

	def __init__(self, args):
		# call super
		super(AbstractTreeLanguageModel, self).__init__(args)

	# overwrite build_model
	def _build_model(self):
		
		# build criterion
		criterion = TreelangCrossEntropyLoss(ntokens=self.ntokens, distance='eucl', temp=self.args.temperature)

		# build model
		model = RNNModel(self.args.model, self.ntokens, self.args.emsize, self.args.nhid, self.args.nlayers, self.args.dropout,
							self.args.dropouth, self.args.dropouti, self.args.dropoute, self.args.wdrop, self.args.tied)
		
		# if resume, load model
		if self.args.resume:
			print('Resuming model ...')
			model_load(self.args.resume)
			self.optimizer.param_groups[0]['lr'] = self.args.lr
			self.model.dropouti, self.model.dropouth, self.model.dropout, self.args.dropoute = self.args.dropouti, self.args.dropouth, self.args.dropout, self.args.dropoute
			
		# apply weight drop
		if self.args.wdrop:
			from merity.weight_drop import WeightDrop
			for rnn in model.rnns:
				if type(rnn) == WeightDrop: rnn.dropout = self.args.wdrop
				elif rnn.zoneout > 0: rnn.zoneout = self.args.wdrop

		# apply cuda
		if self.args.cuda:
			model = model.cuda()
			criterion = criterion.cuda()

		return model, criterion

'''
	A model used for tiny languages built by following all paths starting from the root of a tree.
	Since the datasets tend to be extremely small, we have to iterate over all of the data every epoch.
'''
class TinyTreeLanguageModel(AbstractTreeLanguageModel):

	def __init__(self, args):
		# call super
		super(TinyTreeLanguageModel, self).__init__(args)


	def _evaluate(self, data_source, batch_size=1, dump_vars=None):
		'''
			for the tiny data set, we iterate over the all of the data every epoch
		'''

		# Turn on evaluation mode which disables dropout.
		self.model.eval()
		if self.args.model == 'QRNN': self.model.reset()
		total_loss = 0
		ntokens = self.ntokens
		len_data_source = 0

		# if we dump contexts, need a list to store them
		if not dump_vars is None: contexts = []

		# iterate over sequences of same length
		for seq_len, seq_data in data_source.items():
			for i in range(0, seq_data.size(0) - 1, seq_len):

				# new sequence -> reset hidden state
				hidden = self.model.init_hidden(batch_size)

				# get batch
				data, targets = get_batch(seq_data, i, self.args, seq_len=seq_len, evaluation=True)

				# evaluate
				output, new_hidden = self.model(data, hidden)

				# need to augment output and targets with initial hidden state
				output = output.view(seq_len-1, self.batch_size, self.args.nhid)
				output = torch.cat((hidden[0], output), dim=0)
				targets = torch.cat((data[0].view(1), targets))

				hidden = new_hidden
				total_loss += len(data) * self.criterion(self.model, output, targets).data
				hidden = repackage_hidden(hidden)

				# collect context vectors
				if not dump_vars is None: contexts.append(output)

			len_data_source += len(seq_data)

		# dump contexts
		if not dump_vars is None: dump_contexts(contexts, bsz=batch_size, **dump_vars)

		# return loss
		return total_loss.item() / len_data_source


	def _train(self, epoch):
		'''
			for the tiny data set, we iterate over the all of the data before calling 
			optimizer.step to avoid jerky sgd behaviour
		'''
		
		# Turn on training mode which enables dropout.
		if self.args.model == 'QRNN':self.model.reset()
		total_loss = 0.
		start_time = time.time()
		ntokens = self.ntokens
		batch = 0

		# iterate over sequences of same length
		items = list(self.train_data.items())
		random.shuffle(items)

		# reset gradients of optimizer
		self.optimizer.zero_grad()

		for seq_len, seq_data in items:
			for i in range(0, seq_data.size(0) - 1, seq_len):

				# new sequece -> reset hidden state
				hidden = self.model.init_hidden(self.batch_size)
				lr2 = self.optimizer.param_groups[0]['lr']
				self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.args.bptt
				self.model.train()

				data, targets = get_batch(seq_data, i, self.args, seq_len=seq_len)

				# Starting each batch, we detach the hidden state from how it was previously produced.
				# If we didn't, the model would try backpropagating all the way to start of the dataset.
				hidden = repackage_hidden(hidden)

				output, new_hidden, rnn_hs, dropped_rnn_hs = self.model(data, hidden, return_h=True)

				# need to augment output and targets with initial hidden state
				output = output.view(seq_len-1, self.batch_size, self.args.nhid)
				output = torch.cat((hidden[0], output), dim=0)
				targets = torch.cat((data[0].view(1), targets))

				hidden = new_hidden
				raw_loss = self.criterion(self.model, output, targets)

				loss = raw_loss
				# Activiation Regularization
				if self.args.alpha: loss = loss + sum(self.args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
				# Temporal Activation Regularization (slowness)
				if self.args.beta and seq_len > 2: loss = loss + sum(self.args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
				total_loss += loss

				self.optimizer.param_groups[0]['lr'] = lr2

		total_loss.backward()
	            
		# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs
		if self.args.clip: torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
		self.optimizer.step()


'''
	A model used for small languages built by following all paths from the root of a tree.
	Datasets tend to be larger than for the tiny languages (about 1000 sentences).
'''
class SmallTreeLanguageModel(AbstractTreeLanguageModel):


	def __init__(self, args):
		# call super
		super(SmallTreeLanguageModel, self).__init__(args)


	def _evaluate(self, data_source, batch_size=1, dump_vars=None):
		'''
			Here, we iterate over the data in batches.
		'''

		# Turn on evaluation mode which disables dropout.
		self.model.eval()
		if self.args.model == 'QRNN': self.model.reset()
		total_loss = 0
		ntokens = self.ntokens
		len_data_source = 0

		# if we dump contexts, need a list to store them
		if not dump_vars is None: contexts = []

		# iterate over sequences of same length
		for seq_len, seq_data in data_source.items():
			for i in range(0, seq_data.size(0) - 1, seq_len):

				# new sequence -> reset hidden state
				hidden = self.model.init_hidden(batch_size)

				# get batch
				data, targets = get_batch(seq_data, i, self.args, seq_len=seq_len, evaluation=True)

				# evaluate
				output, new_hidden = self.model(data, hidden)

				# need to augment output and targets with initial hidden state
				output = output.view(seq_len-1, self.batch_size, self.args.nhid)
				output = torch.cat((hidden[0], output), dim=0)
				targets = torch.cat((data[0].view(1), targets))

				hidden = new_hidden
				total_loss += len(data) * self.criterion(self.model, output, targets).data
				hidden = repackage_hidden(hidden)

				# collect context vectors
				if not dump_vars is None: contexts.append(output)

			len_data_source += len(seq_data)

		# dump contexts
		if not dump_vars is None: dump_contexts(contexts, bsz=batch_size, **dump_vars)

		# return loss
		return total_loss.item() / len_data_source


	def _train(self, epoch):
		'''
			Update gradients after each batch
		'''
		
		# Turn on training mode which enables dropout.
		if self.args.model == 'QRNN':self.model.reset()
		total_loss = 0.
		start_time = time.time()
		ntokens = self.ntokens
		batch = 0

		# iterate over sequences of same length
		items = list(self.train_data.items())
		random.shuffle(items)

		# reset gradients of optimizer
		self.optimizer.zero_grad()

		for seq_len, seq_data in items:
			for i in range(0, seq_data.size(0) - 1, seq_len):

				# new sequece -> reset hidden state
				hidden = self.model.init_hidden(self.batch_size)
				lr2 = self.optimizer.param_groups[0]['lr']
				self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.args.bptt
				self.model.train()

				data, targets = get_batch(seq_data, i, self.args, seq_len=seq_len)

				# Starting each batch, we detach the hidden state from how it was previously produced.
				# If we didn't, the model would try backpropagating all the way to start of the dataset.
				hidden = repackage_hidden(hidden)

				output, new_hidden, rnn_hs, dropped_rnn_hs = self.model(data, hidden, return_h=True)

				if self.args.loss == 'treelang_eucl':
					# need to augment output and targets with initial hidden state
					output = output.view(seq_len-1, self.batch_size, self.args.nhid)
					output = torch.cat((hidden[0], output), dim=0)
					targets = torch.cat((data[0].view(1), targets))

				hidden = new_hidden
				raw_loss = self.criterion(self.model, output, targets)

				loss = raw_loss
				# Activiation Regularization
				if self.args.alpha: loss = loss + sum(self.args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
				# Temporal Activation Regularization (slowness)
				if self.args.beta and seq_len > 2: loss = loss + sum(self.args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
				total_loss += loss

				self.optimizer.param_groups[0]['lr'] = lr2

				total_loss.backward()
	            
				# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs
				if self.args.clip: torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
				self.optimizer.step()

				total_loss = 0.
				batch += 1


'''
	This is the general language model used for NL problems.
'''
class TreeLanguageModel(AbstractTreeLanguageModel):

	def __init__(self, args):
		# call super
		super(TreeLanguageModel, self).__init__(args)


	def _evaluate(self, data_source, batch_size=10, dump_vars=None):

		# Turn on evaluation mode which disables dropout.
		self.model.eval()
		if self.args.model == 'QRNN': self.model.reset()
		total_loss = 0
		ntokens = self.ntokens
		hidden = self.model.init_hidden(batch_size)
		for i in range(0, data_source.size(0) - 1, self.args.bptt):
			data, targets = get_batch(data_source, i, self.args, evaluation=True)
			output, hidden = self.model(data, hidden)
			total_loss += len(data) * self.criterion(self.model, output, targets).data
			hidden = repackage_hidden(hidden)
		return total_loss.item() / len(data_source)


	def _train(self, epoch):

		# Turn on training mode which enables dropout.
		if self.args.model == 'QRNN': self.model.reset()
		total_loss = 0
		start_time = time.time()
		ntokens = self.ntokens
		hidden = self.model.init_hidden(self.args.batch_size)
		batch, i = 0, 0
		while i < self.train_data.size(0) - 1 - 1:
			bptt = self.args.bptt if np.random.random() < 0.95 else self.args.bptt / 2.
			# Prevent excessively small or negative sequence lengths
			seq_len = max(5, int(np.random.normal(bptt, 5)))
			# There's a very small chance that it could select a very long sequence length resulting in OOM
			# seq_len = min(seq_len, args.bptt + 10)

			lr2 = self.optimizer.param_groups[0]['lr']
			self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.args.bptt
			self.model.train()
			data, targets = get_batch(self.train_data, i, self.args, seq_len=seq_len)

			# Starting each batch, we detach the hidden state from how it was previously produced.
			# If we didn't, the model would try backpropagating all the way to start of the dataset.
			hidden = repackage_hidden(hidden)
			self.optimizer.zero_grad()

			output, hidden, rnn_hs, dropped_rnn_hs = self.model(data, hidden, return_h=True)
			raw_loss = self.criterion(self.model, output, targets)

			loss = raw_loss
			# Activiation Regularization
			if self.args.alpha: loss = loss + sum(self.args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
			# Temporal Activation Regularization (slowness)
			if self.args.beta: loss = loss + sum(self.args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
			loss.backward()

			# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
			if self.args.clip: torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
			self.optimizer.step()

			total_loss += raw_loss.data
			self.optimizer.param_groups[0]['lr'] = lr2
			if batch % self.args.log_interval == 0 and batch > 0:
				cur_loss = total_loss.item() / self.args.log_interval
				elapsed = time.time() - start_time
				print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
					'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
					epoch, batch, len(self.train_data) // self.args.bptt, self.optimizer.param_groups[0]['lr'],
					elapsed * 1000 / self.args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
				total_loss = 0
				start_time = time.time()
			###
			batch += 1
			i += seq_len

