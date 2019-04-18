import time
import math
import numpy as np
import torch
import torch.nn as nn
import random

from merity.utils import repackage_hidden
from merity.model import RNNModel
from loss.splitnscriterion import SplitNegativeSampleCriterion, SplitCrossEntropy
from treelang.ns_sample import SplitNegativeSampler
from visualize.dump import dump_contexts

class SplitNSLM():

	def __init__(self, args):

		# store args
		self.args = args
		self.asgd = args.asgd

		# splits
		self.splits = [0] + args.splits + [100 * 1000000]
		self.nsplits = len(self.splits) - 1

		# empty list to store val loss
		self.val_loss = []

		# batchsize is 1 for tiny language
		self.batch_size = 1
		self.eval_batch_size = 1
		self.test_batch_size = 1

		# initialize data sets
		self.corpus, self.train_data, self.val_data, self.test_data = self._load_data()
		self.ntokens = len(self.corpus.dictionary) + self.nsplits - 1	# extra tokens for tombstones
		self.ntokens_wots = len(self.corpus.dictionary)

		# initialize model and criterion
		self.optimizer = None
		self.model, self.train_criterion, self.eval_criterion = self._build_model()

		# collect all parameters
		self.params = list(self.model.parameters())
		self.total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in self.params if x.size())
		print('Args:', self.args)
		print('Model total parameters:', self.total_params)

		# initialize optimizer
		if self.optimizer is None:
			if self.args.optimizer == 'sgd':
				self.optimizer = torch.optim.SGD(self.params, lr=self.args.lr, weight_decay=self.args.wdecay)
			elif self.args.optimizer == 'adam':
				self.optimizer = torch.optim.Adam(self.params, lr=self.args.lr, weight_decay=self.args.wdecay)
			else:
				raise ValueError('Wrong optimizer specified! Must be in [sgd, adam].')

		# sampler (update frequencies first)
		self.frequencies = torch.zeros(self.ntokens)
		self.frequencies[:self.ntokens_wots] = self.corpus.frequencies
		for i in range(1, self.nsplits):
			tombstone = self.ntokens_wots + i - 1
			left, right = self.splits[i], min(self.splits[i+1], self.ntokens_wots)
			self.frequencies[tombstone] = torch.sum(self.frequencies[left:right])
		self.sampler = SplitNegativeSampler(self.args.nsamples, self.frequencies, self.splits)

		# finish up
		print('Initialization successful!')

	def _evaluate(self, data_source, batch_size=1, dump_vars=None):

		total_loss = 0
		reset_hidden = True
		self.model.eval()

		# if we dump contexts, need a list to store them
		if not dump_vars is None: contexts = []

		for i in range(data_source.size(0)):

			# control variables
			if reset_hidden:
				# if eos, reset hidden state
				hidden = self.model.init_hidden(batch_size)
				if not dump_vars is None:
					if i > 0: contexts.append(context)
					context = hidden[0][0][0].view(1,-1)
				hidden = repackage_hidden(hidden)

			# get current word and target
			target = data_source[i]

			# apply model to all the words, no splits atm!
			raw_loss, output = self.eval_criterion(self.model, target, hidden[0][0])
			
			# update hidden
			# we want the target hidden state
			hidden = [output[target].view(1, batch_size, -1)]
			if not dump_vars is None: context = torch.cat((hidden[0][0][0].view(1,-1), context), 0)

			# update loss
			total_loss += raw_loss

			# update control variables
			reset_hidden = True if target.data.cpu().numpy()[0] in self.corpus.reset_idxs else False
			# TODO: add other reset conditions

		# dump contexts
		if not dump_vars is None: dump_contexts(contexts, bsz=batch_size, **dump_vars)

		return total_loss.item() / data_source.size(0)

	def _train(self, epoch):

		total_loss, loss = 0, 0
		reset_hidden = True

		for i in range(self.train_data.size(0)):

			# set learning rate and model trainable
			lr2 = self.optimizer.param_groups[0]['lr']
			self.optimizer.param_groups[0]['lr'] = lr2 * 1 / self.args.bptt
			self.model.train()

			# control variables
			if reset_hidden:
				# if eos, reset hidden state
				hidden = self.model.init_hidden(self.batch_size)
				hidden = repackage_hidden(hidden[0])

			bptt = self.args.bptt if np.random.random() < 0.95 else self.args.bptt / 2.
			if i > 0 and i % bptt == 0:
				# all bptt iterations, do optimizer step
				loss.backward()

				# clip and update
				if self.args.clip: torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
				self.optimizer.step()
				self.optimizer.zero_grad()

				# reset loss and hiddens
				hidden = repackage_hidden(hidden[0])
				total_loss = total_loss + loss
				loss = 0

			# determine whether target is in head or not
			target = self.train_data[i]
			target_in_head = target < self.splits[1]

			# if the target is in the head we only have to apply our model once
			if target_in_head:
				
				# take current word, sample negatives, evaluate at once, compute loss
				neg = self.sampler(0, self.args.cuda)

				# reshape pos, negs, and hidden correctly
				data_in, hidden_in = self._posneg2input(target, neg, hidden)
				output, new_hidden, rnn_hs, dropped_rnn_hs = self.model(data_in, hidden_in, return_h=True)

				hiddens = [new_hidden[0][0]]
				outputs = [output]

			# otherwise we have to sample negatives for the tombstone and the actual token
			else:

				# find index of tombstone
				i, tombstone = 0, self.ntokens - self.nsplits + 1
				while self.splits[i+1] <= target: i = i+1

				# sample negatives for tombstone and run model
				tombstone = torch.LongTensor([tombstone + i - 1])
				neg = self.sampler(0, self.args.cuda)
				data_in, hidden_in = self._posneg2input(tombstone, neg, hidden)
				output_ts, hidden_ts, rnn_hs, dropped_rnn_hs = self.model(data_in, hidden_in, return_h=True)

				# udpate hidden
				hidden = hidden_ts[0][0][0].view(1, self.batch_size, -1)

				# now, sample index for actual token and run model
				neg = self.sampler(i, self.args.cuda)
				data_in, hidden_in = self._posneg2input(target, neg, hidden)
				output_tar, new_hidden, rnn_hs, dropped_rnn_hs = self.model(data_in, hidden_in, return_h=True)

				# collect outputs
				hiddens = [hidden_ts[0][0], new_hidden[0][0]]
				outputs = [output_ts, output_tar]


			raw_loss = self.train_criterion(hiddens, output)
			
			# update hidden
			# sample at index 0 is the positive sample
			hidden = new_hidden[0][0][0].view(1, self.batch_size, -1)	

			# regularizer
			loss = loss + raw_loss
			if self.args.alpha: loss = loss + sum(self.args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])

			# update control variables
			reset_hidden = True if target.data.cpu().numpy()[0] in self.corpus.reset_idxs else False
			self.optimizer.param_groups[0]['lr'] = lr2# TODO: add other reset conditions

		# final weight update	
		loss.backward()

		if self.args.clip: torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
		self.optimizer.step()
		self.optimizer.zero_grad()


	def _posneg2input(self, pos, neg, hidden):
		data_in = torch.zeros((1, len(neg)+1))
		data_in[0][0] = pos
		data_in[0][1:] = neg
		hidden_in = hidden[0].repeat(1, self.args.nsamples+1, 1)
		return data_in.cuda(), [hidden_in]

	def _model_load(self, fn):
		with open(fn, 'rb') as f:
			self.model, self.train_criterion, self.eval_criterion, self.optimizer = torch.load(f)

	def _model_save(self, fn):
		with open(fn, 'wb') as f:
			torch.save([self.model, self.train_criterion, self.eval_criterion, self.optimizer], f)

	def _load_data(self):

		# imports
		import os
		import hashlib
		from merity.utils import batchify
		import treelang.ns_data as data

		# hash
		fn = 'corpus.{}.data'.format(hashlib.md5(self.args.data.encode()).hexdigest())
		if os.path.exists(fn):
			print('Loading cached dataset...')
			corpus = torch.load(fn)
		else:
			print('Producing dataset...')
			corpus = data.Corpus(self.args.data) # need data which stores indices of <eos>, <eol>, etc
			torch.save(corpus, fn)

		# batchify
		train_data = batchify(corpus.train, self.batch_size, self.args)
		val_data = batchify(corpus.valid, self.eval_batch_size, self.args)
		test_data = batchify(corpus.test, self.test_batch_size, self.args)

		return corpus, train_data, val_data, test_data

	def _build_model(self):

		# build criterion (negative sampling crit)
		train_criterion = SplitNegativeSampleCriterion(self.args.temp)
		eval_criterion = SplitCrossEntropy(self.ntokens, self.args.temp)

		# build model
		ntokens = self.ntokens
		model = RNNModel(self.args.model, ntokens, self.args.emsize, self.args.nhid, self.args.nlayers, self.args.dropout,
							self.args.dropouth, self.args.dropouti, self.args.dropoute, self.args.wdrop, self.args.tied)
		
		# if resume, load model
		if self.args.resume:
			print('Resuming model ...')
			model, criterion = self._model_load(self.args.resume)
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
			train_criterion = train_criterion.cuda()
			eval_criterion = eval_criterion.cuda()

		return model, train_criterion, eval_criterion

	def train(self):
		'''
			trains the language model for args.epochs number of epochs and evaluates
			it on the test data. returns best loss over all iterations and loss of 
			saved model.
		'''
		
		# get learning rate and initialize loss to high number
		lr = self.args.lr
		stored_loss = 1e8
		best_val_loss = []
		
		# iterate over epochs
		for epoch in range(1, self.args.epochs + 1):

			#self.eval_criterion.temp, self.train_criterion.temp = 60 / epoch, 60/epoch# train
			epoch_start_time = time.time()
			self._train(epoch)

			# this is if asgd is active
			if 't0' in self.optimizer.param_groups[0]:
				tmp = {}
				for prm in self.model.parameters():
					tmp[prm] = prm.data.clone()
					if 'ax' in self.optimizer.state[prm]:
						prm.data = self.optimizer.state[prm]['ax'].clone()

				val_loss2 = self._evaluate(self.val_data, self.eval_batch_size)
				print('-' * 89)
				print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
					'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
						epoch, (time.time() - epoch_start_time), val_loss2, val_loss2, val_loss2 / math.log(2)))
				print('-' * 89)

				if val_loss2 < stored_loss:
					self._model_save(self.args.save)
					print('Saving Averaged!')
					stored_loss = val_loss2

				for prm in self.model.parameters():
					prm.data = tmp[prm].clone()

				self.val_loss.append(val_loss2)

			# otherwise
			else:
				val_loss = self._evaluate(self.val_data, self.eval_batch_size)
				print('-' * 89)
				print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
					'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
					epoch, (time.time() - epoch_start_time), val_loss, val_loss, val_loss / math.log(2)))
				print('-' * 89)

				if val_loss < stored_loss:
					self._model_save(self.args.save)
					print('Saving model (new best validation)')
					stored_loss = val_loss

				if self.args.optimizer == 'sgd' and 't0' not in self.optimizer.param_groups[0] and (len(best_val_loss)>self.args.nonmono and val_loss > min(best_val_loss[:-self.args.nonmono])):
					if self.asgd:
						print('Switching to ASGD')
						self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.args.lr, t0=0, lambd=0., weight_decay=self.args.wdecay)

				if epoch in self.args.when:
					print('Saving model before learning rate decreased')
					self._model_save('{}.e{}'.format(self.args.save, epoch))
					print('Dividing learning rate by 10')
					self.optimizer.param_groups[0]['lr'] /= 10.

				best_val_loss.append(val_loss)
				self.val_loss.append(val_loss)

			# every dumpat iteration: store contexts to file for later plotting
			if self.args.dumpat > 0 and epoch % self.args.dumpat == 0:
				dump_vars = dict({'basepath': self.args.dumpto, 'epoch':epoch, 'hsz':self.args.nhid})
				self._evaluate(self.test_data, self.test_batch_size, dump_vars)


		# done with training!
		# load best model and evaluate it on the test set
		self._model_load(self.args.save)
		test_loss = self._evaluate(self.test_data, self.test_batch_size)
		print('=' * 89)
		print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
			test_loss, math.exp(test_loss), test_loss / math.log(2)))
		print('=' * 89)
		print('| End of training | best loss {:5.2f} | best ppl {:8.2f} | best bpc {:8.3f}'.format(
			stored_loss, math.exp(stored_loss), stored_loss / math.log(2)))
		print('=' * 89)

		return test_loss



