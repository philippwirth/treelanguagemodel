import random
import torch
import time
import numpy as np
import itertools

from operator import mul


def args_to_dict(args):
	result = dict()
	result['lr'] = args.lr
	result['kernel'] = args.kernel
	result['optimizer'] = args.optimizer
	result['when'] = args.when
	result['temperature'] = args.temperature
	result['asgd'] = args.asgd
	return result


def gridsearch_step(args, learning_rates, kernels, optimizers, whens, temps, asgds, tiny=True, treelang=True, K=1):

	if args.lmodel == 'tiny':
		from language_models.tiny_language_model import TinyLanguageModel as LanguageModel
	elif args.lmodel == 'small':
		from language_models.small_language_model import SmallLanguageModel as LanguageModel
	elif args.lmodel == 'regular':
		from language_models.language_model import LanguageModel
	else:
		raise ValueError("invalid argument: lmodel. Needs to be in [treelang_tiny, treelang_small, general]")

	# some constant stuff
	args.dumpat = 0

	# grid
	L = [learning_rates, kernels, optimizers, whens, temps, asgds]

	# some info
	n_settings = np.prod([len(l) for l in L])
	print('Doing gridsearch over ' + str(n_settings) + ' settings.')
	L = list(itertools.product(*L))

	# variables for storage
	best_loss, best_settings = 1e5, dict()

	# do the grid
	for (lr, kernel, optimizer, when, temp, asgd) in L:

		# reset seed for reproducibility.
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		if torch.cuda.is_available():
			if not args.cuda:
				print("WARNING: You have a CUDA device, so you should probably run with --cuda")
			else:
				torch.cuda.manual_seed(args.seed)

		# fix settings
		args.lr = lr
		args.kernel = kernel
		args.optimizer = optimizer
		args.when = when
		args.temperature = temp
		args.asgd = asgd

		cur_loss = np.zeros(K)
		for k in range(K):
			lm = LanguageModel(args)
			cur_loss[k] = lm.train()
		cur_loss = np.amin(cur_loss)

		if cur_loss < best_loss:
			best_loss = cur_loss
			best_settings = args_to_dict(args)

	return best_loss, best_settings

def gridsearch_treelang(args):

	if args.lmodel == 'tiny':
		from language_models.tiny_language_model import TinyLanguageModel as LanguageModel
	elif args.lmodel == 'small':
		from language_models.small_language_model import SmallLanguageModel as LanguageModel
	elif args.lmodel == 'regular':
		from language_models.language_model import LanguageModel
	else:
		raise ValueError("invalid argument: lmodel. Needs to be in [treelang_tiny, treelang_small, general]")

	models = ['RNN', 'GRU']
	learning_rates = [0.001 * pow(2, i) for i in range(16)]
	optimizers = ['adam', 'sgd']
	whens = [[-1], [25], [25, 35], [50]]
	asgds = [True, False]
	temps = [pow(2,i) for i in range(12)]
	kernels = ['polynomial1', 'polynomial2', 'dot']

	# store best settings over #epochs for each model
	storage = dict()
	for model in models:

		args.model = model

		# do gridsearch step
		best_loss, best_settings = gridsearch_step(args, model, learning_rates, kernels, optimizers, whens, temps, asgds)
		storage[model] = [best_loss, best_settings]


	# store best result over 1000 epochs for each model
	args.epochs = 1000
	for model in models:

		args.model = model

		# reset seed for reproducibility.
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		if torch.cuda.is_available():
			if not args.cuda:
				print("WARNING: You have a CUDA device, so you should probably run with --cuda")
			else:
				torch.cuda.manual_seed(args.seed)

		# fix settings
		settings = storage[model][1]
		args.lr = settings['lr']
		args.kernel = settings['kernel']
		args.optimizer = settings['optimizers']
		args.when = settings['when']
		args.temperature = settings['temperature']
		args.asgd = settings['asgd']

		# run lm
		lm = LanguageModel(args)
		loss = lm.train()

		# store
		storage[model].append(loss)

	print(storage)



