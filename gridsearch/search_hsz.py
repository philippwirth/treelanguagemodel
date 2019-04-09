import argparse
import random
import torch
import time
import numpy as np

import itertools

def search_hsz(args):

	# import correct language model
	if args.lmodel == 'tiny':
		from language_models.tiny_language_model import TinyLanguageModel as LanguageModel
	elif args.lmodel == 'small':
		from language_models.small_language_model import SmallLanguageModel as LanguageModel
	elif args.lmodel == 'regular':
		from language_models.language_model import LanguageModel
	else:
		raise ValueError("invalid argument: lmodel. Needs to be in [treelang_tiny, treelang_small, general]")

	# initialize some lists
	models = ['GRU', 'LSTM']
	losses = ['treelang', 'splitcross']
	hszs = [2, 8, 32, 128]		# different hidden sizes
	lrs = [0.01, 0.1]	# different learning rates
	L = list(itertools.product(*[hszs, lrs]))

	storage = dict()
	for model, loss in zip(models, losses):

		storage[model] = []
		args.model = model
		args.loss = loss
		for (hsz, lr) in L:

			# set the random seed manually for reproducibility.
			random.seed(args.seed)
			np.random.seed(args.seed)
			torch.manual_seed(args.seed)
			if torch.cuda.is_available():
				if not args.cuda:
					print("WARNING: You have a CUDA device, so you should probably run with --cuda")
				else:
					torch.cuda.manual_seed(args.seed)

			# set args and build language model
			args.hsz = hsz
			args.emsize = hsz
			args.lr = lr
			lm = LanguageModel(args)
			loss = lm.train()

			# update storage
			storage[model].append([hsz, lr, loss])

	print(storage)
	print('hsz-search done.')

def search_temp(args):

	# import correct language model
	if args.lmodel == 'tiny':
		from language_models.tiny_language_model import TinyLanguageModel as LanguageModel
	elif args.lmodel == 'small':
		from language_models.small_language_model import SmallLanguageModel as LanguageModel
	elif args.lmodel == 'regular':
		from language_models.language_model import LanguageModel
	else:
		raise ValueError("invalid argument: lmodel. Needs to be in [treelang_tiny, treelang_small, general]")

	# initialize some lists
	models = ['RNN', 'GRU']
	losses = ['treelang', 'treelang']
	temps = [10*i for i in range(1,51)]		# different hidden sizes
	lrs = [0.01, 0.05, 0.1, 0.15]	# different learning rates

	for model, loss in zip(models, losses):
	
		loss_by_temp = np.zeros(len(temps))
		args.model = model
		args.loss = loss
		for i, temp in enumerate(temps):

			args.temperature = temp
			best_loss = 1e5
			for lr in lrs:

				# set the random seed manually for reproducibility.
				random.seed(args.seed)
				np.random.seed(args.seed)
				torch.manual_seed(args.seed)
				if torch.cuda.is_available():
					if not args.cuda:
						print("WARNING: You have a CUDA device, so you should probably run with --cuda")
					else:
						torch.cuda.manual_seed(args.seed)

				# set args and build language model
				args.lr = lr
				lm = LanguageModel(args)
				loss = lm.train()

				if loss < best_loss:
					best_loss = loss

			loss_by_temp[i] = best_loss

		np.savetxt('loss_by_temp_' + model + '.txt', loss_by_temp, delimiter=' ')

	print('search-temp done.')

		
