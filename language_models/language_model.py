import time
import math
import numpy as np
import torch
import torch.nn as nn
import random

# super
from language_models.abstract_language_model import AbstractLanguageModel

# utils
from merity.utils import get_batch, repackage_hidden
from visualize.dump import dump_contexts

'''
	This is the general language model used for NL problems.
'''
class LanguageModel(AbstractLanguageModel):

	def __init__(self, args):
		# call super
		super(LanguageModel, self).__init__(args)


	def _train(self, epoch):

		# Turn on training mode which enables dropout.
		if self.args.model == 'QRNN': self.model.reset()
		total_loss = 0
		start_time = time.time()
		ntokens = self.ntokens
		hidden = self.model.init_hidden(self.args.batch_size)
		batch, i = 0, 0
		while i < self.train_data.size(0) - 1 - 1:


			# test!
			#self.model.eval()

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

			if self.args.loss == 'treelang':
				# need to augment output and targets with initial hidden state
				output = torch.cat((hidden[0][0], output), dim=0)
				targets = torch.cat((data[0], targets))

			total_loss += len(data) * self.criterion(self.model, output, targets).data
			hidden = repackage_hidden(hidden)
		return total_loss.item() / len(data_source)

