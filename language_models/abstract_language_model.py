import time
import math
import numpy as np
import torch
import torch.nn as nn
import random

# imports from treelang!
import merity.data as data
from merity.model import RNNModel

# same same
from merity.utils import batchify, get_batch, repackage_hidden

class AbstractLanguageModel():

	def __init__(self, args):

		# store args
		self.args = args
		self.asgd = args.asgd

		# empty list to store val loss
		self.val_loss = []

		# batchsize is 1 for tiny language
		self.batch_size = args.batch_size
		self.eval_batch_size = 1
		self.test_batch_size = 1

		# initialize data sets
		self.corpus, self.train_data, self.val_data, self.test_data = self._load_data()
		self.ntokens = len(self.corpus.dictionary)

		# initialize model and criterion
		self.model, self.criterion = self._build_model()

		# collect all parameters
		self.params = list(self.model.parameters()) + list(self.criterion.parameters())
		self.total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in self.params if x.size())
		print('Args:', self.args)
		print('Model total parameters:', self.total_params)

		# initialize optimizer
		if self.args.optimizer == 'sgd':
			self.optimizer = torch.optim.SGD(self.params, lr=self.args.lr, weight_decay=self.args.wdecay)
		if self.args.optimizer == 'adam':
			self.optimizer = torch.optim.Adam(self.params, lr=self.args.lr, weight_decay=self.args.wdecay)

		# finish up
		print('Initialization successful!')

	@abstractmethod
	def _evaluate(self, data_source, batch_size=10, dump_vars=None):
		pass

	@abstractmethod
	def _train(self, epoch):
		pass

	def _model_load(self, fn):
		with open(fn, 'rb') as f:
			self.model, self.criterion, self.optimizer = torch.load(f)

	def _model_save(self, fn):
		with open(fn, 'wb') as f:
			torch.save([self.model, self.criterion, self.optimizer], f)

	def _load_data(self):

		# imports
		import os
		import hashlib

		# hash
		fn = 'corpus.{}.data'.format(hashlib.md5(self.args.data.encode()).hexdigest())
		if os.path.exists(fn):
			print('Loading cached dataset...')
			corpus = torch.load(fn)
		else:
			print('Producing dataset...')
			corpus = data.Corpus(self.args.data)
			torch.save(corpus, fn)

		# need to batchify differently for the treelang data
		train_data = batchify(corpus.train, self.batch_size, self.args)
		val_data = batchify(corpus.valid, self.eval_batch_size, self.args)
		test_data = batchify(corpus.test, self.test_batch_size, self.args)

		return corpus, train_data, val_data, test_data

	def _build_model(self):
		
		# build criterion
		if self.args.loss == 'splitcross':
			from merity.splitcross import SplitCrossEntropyLoss
			criterion = None
		elif self.args.loss == 'treelang_eucl':
			from treelang.crossentropy import TreelangCrossEntropyLoss
			criterion = TreelangCrossEntropyLoss(ntokens=self.ntokens, distance='eucl', temp=self.args.temperature)
		else:
			raise ValueError("args.loss must be in ['splitcross', 'treelang_eucl']")

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

		# split tokens for quick crossentropy	
		if not criterion:
			splits = []
			if self.ntokens > 500000:
				# One Billion
				# This produces fairly even matrix mults for the buckets:
				# 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
				splits = [4200, 35000, 180000]
			elif self.ntokens > 75000:
				# WikiText-103
				splits = [2800, 20000, 76000]
			print('Using', splits)
			criterion = SplitCrossEntropyLoss(self.args.emsize, splits=splits, verbose=False)

		# apply cuda
		if self.args.cuda:
			model = model.cuda()
			criterion = criterion.cuda()

		return model, criterion

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

			# train
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

