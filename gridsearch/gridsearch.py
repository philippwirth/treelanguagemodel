import random
import torch
import time
import numpy as np
import itertools

from operator import mul


def args_to_dict(args, asgd):
	result = dict()
	result['lr'] = args.lr
	result['alpha'] = args.alpha
	result['beta'] = args.beta
	result['optimizer'] = args.optimizer
	result['when'] = args.when
	result['temperature'] = args.temperature
	result['asgd'] = args.asgd
	return result


def gridsearch_step(args, model, epochs,
				learning_rates, alphas, betas, optimizers, whens, temps, asgds,
				tiny=True, treelang=True, K=1):

	# some constant stuff
	args.dumpat = 0
	args.tiny = tiny
	args.treelang = True
	args.epochs = epochs

	# model
	args.model = model
	if args.lmodel == 'treelangtiny':
	    from language_models.tree_language_model import TinyTreeLanguageModel as LanguageModel
	elif args.lmodel == 'treelangsmall':
	    from language_models.tree_language_model import SmallTreeLanguageModel as LanguageModel
	elif args.lmodel == 'treelang':
	    from language_models.tree_language_model import TreeLanguageModel as LanguageModel
	elif args.lmodel == 'meritytiny':
	    from language_models.merity_language_model import TinyLanguageModel as LanguageModel
	elif args.lmodel == 'meritysmall':
	    from language_models.merity_language_model import SmallLanguageModel as LanguageModel
	elif args.lmodel == 'merity':
	    from language_models.merity_language_model import LanguageModel
	else:
	    raise ValueError("invalid argument: lmodel. Needs to be in [treelang_tiny, treelang_small, general]")

	# grid
	L = [learning_rates, alphas, betas, optimizers, whens, temps, asgds]

	# some info
	n_settings = np.prod([len(l) for l in L])
	print('Doing gridsearch over ' + str(n_settings) + ' settings.')
	L = list(itertools.product(*L))

	# variables for storage
	best_loss, best_settings = 1e5, dict()

	# do the grid
	for (lr, alpha, beta, optimizer, when, temp, asgd) in L:

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
		args.alpha = alpha
		args.beta = beta
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

def gridsearch(args, treelang):

	# create initial lists
	epochs = args.epochs
	if treelang:
		models = ['RNN', 'GRU']
		temps = [pow(2,i) for i in range(10)]
	else:
		models = ['LSTM']
		temps = [1

	learning_rates = [0.001 * pow(2,i) for i in range(16)]
	alphas = [0, 2]
	betas = [0, 1]
	optimizers = ['adam', 'sgd']
	whens = [[-1], [25], [25, 35], [25, 50], [50]]
	asgds = [True, False]

	storage = dict()
	for model in models:

		# do gridsearch step
		best_loss, best_settings = gridsearch_step(args, model, epochs, learning_rates, alphas, betas, optimizers, whens, temps, asgds)

		#refine settings
		alphas = [best_settings['alpha']]
		betas = [best_settings['beta']]
		optimizers = [best_settings['optimizer']]
		whens = [best_settings['when']] + [[100], [100, 250]]
		temps = [i for i in range(best_settings['temperature'] // 2, 2*best_settings['temperature'], 10)]
		asgds = [best_settings['asgd']]

		# refine learning rates
		left = best_settings['lr'] / 2
		right = best_settings['lr'] * 2
		learning_rates = np.linspace(left, right, 10)

		epochs = epochs * 10
		best_loss, best_settings = gridsearch_step(args, model, epochs, learning_rates, alphas, betas, optimizers, whens, temps, asgds)
		storage[model] = [best_loss, best_settings]


	# output
	for model in models:

		print('Results for ' + model)
		print('Best Loss: ' + str(storage[model][0]))
		print('Best Sets: ' + str(storage[model][1]))





