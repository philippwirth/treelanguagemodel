import time
import math
import numpy as np
import torch
import torch.nn as nn
import random

# super
from language_models.abstract_language_model import AbstractLanguageModel

# utils
from merity.utils import repackage_hidden
from treelang.utils import get_batch_treelang
from visualize.dump import dump_contexts

class TinyLanguageModel(AbstractLanguageModel):

	def __init__(self, args):

		# fix batch_size to 1 (1 sequence processed at a time)
		args.batch_size = 1

		# call super
		super(TinyLanguageModel, self).__init__(args)


	# tiny training function
	def _train(self, epoch):
		'''
			for the tiny data set, we iterate over the all of the data before calling 
			optimizer.step to avoid jerky sgd behaviour
		'''
		
		# Turn on training mode which enables dropout.
		if self.args.model == 'QRNN':self.model.reset()
		total_loss = 0.
		ntokens = self.ntokens

		# iterate over sequences of same length
		items = list(self.train_data.items())
		random.shuffle(items)

		# reset gradients of optimizer
		self.optimizer.zero_grad()

		for seq_len, seq_data in items:
			for i in range(0, seq_data.size(0) - 1, seq_len):

				#self.model.eval()

				# new sequece -> reset hidden state
				hidden = self.model.init_hidden(self.batch_size)
				lr2 = self.optimizer.param_groups[0]['lr']
				self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.args.bptt
				self.model.train()

				data, targets = get_batch_treelang(seq_data, i, self.args, seq_len=seq_len)

				# Starting each batch, we detach the hidden state from how it was previously produced.
				# If we didn't, the model would try backpropagating all the way to start of the dataset.
				hidden = repackage_hidden(hidden)

				output, new_hidden, rnn_hs, dropped_rnn_hs = self.model(data, hidden, return_h=True)

				if self.args.loss == 'treelang':
					# need to augment output and targets with initial hidden state
					output = torch.cat((hidden[0][0], output), dim=0)
					targets = torch.cat((data[0], targets))

				#self.model.train()
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


	# tiny evaluation function
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
				data, targets = get_batch_treelang(seq_data, i, self.args, seq_len=seq_len, evaluation=True)

				# evaluate
				output, new_hidden = self.model(data, hidden)

				if self.args.loss == 'treelang':
					# need to augment output and targets with initial hidden state
					output = torch.cat((hidden[0][0], output), dim=0)
					targets = torch.cat((data[0], targets))

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


