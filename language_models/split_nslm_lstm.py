import time
import math
import numpy as np
import torch
import torch.nn as nn
import random

from merity.utils import repackage_hidden
from merity.model import RNNModel
from loss.splitnscriterion import NegativeSampleCriterion, SplitCrossEntropy
from treelang.sample import NegativeSampler
from treelang.utils import get_sequence
from visualize.dump import dump_contexts, dump
from treelang.eucl_distance import EuclEntailmentCone
class NS_LSTM():

	def __init__(self, args):

		# store args
		self.args = args
		self.asgd = args.asgd

		# empty list to store val loss
		self.val_loss = []

		# batchsize is 1 for tiny language
		self.batch_size = 1
		self.eval_batch_size = 1
		self.test_batch_size = 1

		# initialize data sets
		self.corpus, self.train_data, self.val_data, self.test_data = self._load_data()
		self.ntokens = len(self.corpus.dictionary)

		# initialize model and criterion
		self.optimizer = None
		self.model, self.criterion = self._build_model()

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
		self.frequencies = self.corpus.frequencies
		self.sampler = NegativeSampler(self.args.nsamples, self.frequencies)

		self.dist_fn = nn.PairwiseDistance(p=2)

		# finish up
		print('Initialization successful!')

	def _evaluate(self, data_source, epoch, batch_size=1, dump_vars=None, dump_distance=True, dump_entropy=True):

		total_loss = 0
		reset_hidden = True
		self.model.eval()

		# if we dump contexts, need a list to store them
		if not dump_vars is None: contexts = []
		if dump_distance: distances = []
		if dump_entropy: entropies = []

		for i in range(data_source.size(0)):

			# control variables
			if reset_hidden:
				# if eos, reset hidden state
				if i == 0:
					hidden = self.model.init_hidden(batch_size)
				else:
					hidden = self._reset_hidden(hidden, 1)

				if not dump_vars is None:
					if not dump_vars and i > 0: contexts.append(context)
					context = hidden[0][0][0].view(1,-1)

			# get current word and target
			target = data_source[i]

			# apply model to all the words, no splits atm!
			bsz, tokens = 128, torch.LongTensor([i for i in range(self.ntokens)]).cuda()
			nbatch = len(tokens) // bsz
			nbatch = nbatch if (len(tokens) % bsz) == 0 else nbatch + 1

			outputs = []
			for j in range(nbatch):
				# apply model to all words in the split
				ntokens = bsz if (j+1)*bsz <= len(tokens) else len(tokens) % bsz
				hiddens = self._copy_hidden(hidden, ntokens)					# copy hidden state nbatch times
				token_batch = tokens[j*bsz:j*bsz+ ntokens]		# get batch of words
				output, new_h = self.model(token_batch.view(1,-1), hiddens)	# evaluate
				outputs.append(output)

				# keep hidden
				if j*bsz <= target and j*bsz + bsz > target:
					hss = [new_h[l][0][0][target-j*bsz].view(1, self.batch_size, -1) for l in range(self.args.nlayers)]
					css = [new_h[l][1][0][target-j*bsz].view(1, self.batch_size, -1) for l in range(self.args.nlayers)]
					new_hidden = [(hs, cs) for hs,cs in zip(hss, css)]

			# compute distances between input and outputs
			outputs = torch.cat(outputs, dim=0)
			dist = -self.dist_fn(hidden[self.args.nlayers-1][0][0][0], outputs) + self.model.decoder.bias
			softmaxed = torch.nn.functional.log_softmax(dist + 0.00001, dim=0)
			raw_loss = -softmaxed[target].item()
			
			# update hidden
			# we want the target hidden state
			hidden = new_hidden
			if not dump_vars is None: context = torch.cat((hidden[0][0][0].view(1,-1), context), 0)
			if dump_distance: distances.append(dist[target].item())
			if dump_entropy: entropies.append(raw_loss)

			# update loss
			total_loss += raw_loss / data_source.size(0)

			# update control variables
			reset_hidden = True if target.data.cpu().numpy()[0] in self.corpus.reset_idxs else False
			# TODO: add other reset conditions

		# dump contexts
		if not dump_vars is None: dump_contexts(contexts, bsz=batch_size, **dump_vars)
		if dump_distance: dump(distances, basepath=self.args.dump_distance+'-'+str(epoch))
		if dump_entropy: dump(entropies, basepath=self.args.dump_entropy+'-'+str(epoch))

		return total_loss

	def _train(self, epoch):

		start_time = time.time()
		total_loss, loss = 0, 0

		i, last_update = 0, 0

		# determine bptt
		bptt2 = 0
		lr2 = self.optimizer.param_groups[0]['lr']

		while i < self.train_data.size(0):

			# make model trainable
			self.model.train()

			# initialize hidden to 0
			if i == 0:
				# initialize both hidden state and cell state
				hidden = self.model.init_hidden(self.args.batch_size)
			else:
				# keep cell state and only update hidden state
				hidden = self._reset_hidden(hidden, 1)

			# backpropagate etc!
			if i - last_update >= bptt2:

				# keep note of last update
				last_update = i

				if i > 0:
					# backpropagate 
					loss.backward()
					if self.args.clip: torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
					self.optimizer.step()
				
				# set optimizer gradients to 0
				self.optimizer.zero_grad()

				# determine bptt
				bptt = self.args.bptt if np.random.random() < 0.95 else self.args.bptt / 2.
				bptt2 = max(5, int(np.random.normal(bptt, 5)))

				# set learning rate
				self.optimizer.param_groups[0]['lr'] = lr2
				lr2 = self.optimizer.param_groups[0]['lr']
				self.optimizer.param_groups[0]['lr'] = lr2 * bptt2 / self.args.bptt

				# detach hidden
				hidden = repackage_hidden(hidden)

				# reset loss
				total_loss = total_loss + loss
				loss = 0


			# get the next sequence (a line from the data set atm)
			sequence = get_sequence(self.train_data, i, self.corpus.reset_idxs)

			# determine sequence length, i.e. how many steps we take at a time
			seq_len = 8 if np.random.random() > .5 else 10	# change hardcoded stuff!

			# evaluate positive and negative samples
			outputs = []
			for j in range(0, len(sequence), seq_len):

				# determine the actual sequence length
				act_seq_len = seq_len if j+seq_len <= len(sequence) else len(sequence) % seq_len

				# sample negatives
				# input shape is: act_seq_len x (1 + nsamples*act_seq_len)
				# hiddn shape is: [1 x (1 + nsamples*act_seq_len) x hsz]
				data_in = self.sampler(sequence, j, act_seq_len, self.args.cuda)
				hidden_in = self._copy_hidden(hidden, self.args.nsamples*act_seq_len+1)

				# apply model to all of them
				output, hidden, rnn_hs, dropped_rnn_hs = self.model(data_in, hidden_in, return_h=True)

				# concatenate hidden state with output & append
				output = output.view(act_seq_len, 1+self.args.nsamples*act_seq_len, -1)
				output = torch.cat((hidden_in[0][0], output), 0)

				# apply criterion
				raw_loss = self.criterion(output, data_in, self.model.decoder.bias, self.dist_fn)	

				# regularizer
				loss = loss + raw_loss
				if self.args.alpha: loss = loss + sum(self.args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])

				# update hidden state
				hss = [hidden[l][0][0][0].view(1, self.batch_size, -1) for l in range(self.args.nlayers)]
				css = [hidden[l][1][0][0].view(1, self.batch_size, -1) for l in range(self.args.nlayers)]
				hidden = [(hs, cs) for hs,cs in zip(hss, css)]
			
			i = i + len(sequence)
			#print(100 * i / self.train_data.size(0))

		# final weight update	
		loss.backward()

		if self.args.clip: torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
		self.optimizer.step()

		# reset learning rate
		self.optimizer.param_groups[0]['lr'] = lr2

		return total_loss

	def _reset_hidden(self, hidden, keep):

		keep_states = [hidden[l][keep] for l in range(self.args.nlayers)]
		hidden = self.model.init_hidden(self.args.batch_size)
		hidden = [(hs[0], cs) for hs, cs in zip(hidden, keep_states)]
		return hidden


	def _copy_hidden(self, hidden, n):
		new_hidden = []
		for h in hidden:
			hs = h[0].repeat(1, n, 1)
			cs = h[0].repeat(1, n, 1)
			new_hidden.append((hs, cs))
		return new_hidden

	def _posneg2input(self, pos, neg, hidden):
		data_in = torch.cat((pos,neg)).view(1,-1)
		hidden_in = hidden[0].repeat(1, self.args.nsamples+1, 1)
		return data_in.cuda(), [hidden_in]

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
		criterion = NegativeSampleCriterion(self.args.temp)

		# build model
		ntokens = self.ntokens
		model = RNNModel('LSTM', ntokens, self.args.emsize, self.args.nhid, self.args.nlayers, self.args.dropout,
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

			print(self.model.decoder.bias)

			# train
			epoch_start_time = time.time()
			self._train(epoch)

			if epoch % self.args.evaluate > 0:
				print('Time: ' + str(time.time() - epoch_start_time))
				continue

			# this is if asgd is active
			if 't0' in self.optimizer.param_groups[0]:
				tmp = {}
				for prm in self.model.parameters():
					tmp[prm] = prm.data.clone()
					if 'ax' in self.optimizer.state[prm]:
						prm.data = self.optimizer.state[prm]['ax'].clone()

				val_loss2 = self._evaluate(self.val_data, epoch, batch_size=self.eval_batch_size)
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
				val_loss = self._evaluate(self.val_data, epoch, batch_size=self.eval_batch_size)
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


		# done with training!
		# load best model and evaluate it on the test set
		self._model_load(self.args.save)
		test_loss = self._evaluate(self.test_data, self.args.epochs+1, batch_size=self.test_batch_size)
		print('=' * 89)
		print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
			test_loss, math.exp(test_loss), test_loss / math.log(2)))
		print('=' * 89)
		print('| End of training | best loss {:5.2f} | best ppl {:8.2f} | best bpc {:8.3f}'.format(
			stored_loss, math.exp(stored_loss), stored_loss / math.log(2)))
		print('=' * 89)

		return test_loss



