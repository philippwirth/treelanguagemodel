from networkx import DiGraph, write_gpickle, read_gpickle
from networkx.algorithms.dag import descendants
from networkx.algorithms.shortest_paths.generic import shortest_path
from scipy.stats import powerlaw
import matplotlib.pyplot as plt
from collections import deque
import random
import numpy as np
import math
import csv
import os
import networkx

class TreeLangGenerator:

	def __init__(self, ntokens, depth, mode='uniform', pstop=0.1, lam=2, a=1, k=2.5):

		if ntokens > 676: 	   	# vocab not representable by two letter 26 (TODO)
			raise ValueError("ntokens > 676 (not representable by two letters)")

		self.ntokens = ntokens 	# vocabulary size
		self.depth = depth     	# maximum length for sentence
		self.mode = mode		# either uniform or power law (determines degree of nodes)
		self.pstop = pstop		# probability of a node having no descendants

		self.lam = lam 			# this is for the poisson distribution
		self.a, self.k = a, k 	# this is for the power law stuff

		self.vocab = None	   	# vocabulary
		self.data = None	   	# data (text corpus)
		self.T = None		   	# language tree

		self.nnodes = 0	   		# number of nodes
		self.alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
					 'n', 'o', 'p','q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

	def generate_sentences(self, seed=1111):
		''' take a random seed, build a random tree, and use the tree to generate
			a dataset of sentences. each sentence corresponds to a path from the
			root to any node in the tree. 
		'''
		
		# generate the vocabulary
		self.vocab = self._generate_vocab()

		# generate the tree
		self.T = self._generate_tree(seed)

		# traverse the tree to collect sentences
		self.data = self._collect_sentences()

		# compute lower bound perplexity
		self.ppl = self._perplexity()


	def save(self, base='data/treelang/'):
		''' store tree, info, and data to the folder specified by base '''

		# if the directory doesn't exist, make it
		if not os.path.isdir(base):
			os.mkdir(base)

		# store the graph
		write_gpickle(self.T, base + 'tree.gpickle')
		
		# store info
		info = dict({'ntokens':self.ntokens, 'depth':self.depth, 'mode':self.mode, 'pstop':self.pstop, 'nnodes':self.nnodes, 'ppl':self.ppl, 'alphabet':self.alphabet})
		with open(base + 'info.csv', 'w+') as csv_file:
			writer = csv.writer(csv_file)
			for key, value in info.items():
				writer.writerow([key, value])

		# store the dataset
		extension = '.txt'
		for filename in ['train', 'test', 'valid']:

			path = os.path.join(base, filename + extension)

			# check if the file already exists (we don't want to overwrite)
			if os.path.exists(path):
				print(path + " already exists, continuing...")
				continue

			# if not, write line by line to the file
			with open(path, "w+") as f:
				for line in self.data[filename]:
					f.write(line + '\n')


	def load(self, base='data/treelang/'):
		''' recover tree, info, and data from folder specified by base '''

		with open(base + 'info.csv') as csv_file:
			reader = csv.reader(csv_file)
			info = dict(reader)
		
		self.ntokens = int(info['ntokens'])
		self.depth = int(info['depth'])
		self.mode = info['mode']
		self.pstop =float(info['pstop'])
		self.nnodes = int(info['nnodes'])
		self.alphabet = [info['alphabet'][2 + i*5] for i in range(26)] # wow that's ugly

		self.vocab = self._generate_vocab()
		self.T = read_gpickle(base + 'tree.gpickle')
		self.data = self._collect_sentences()

		self.ppl = self._perplexity()


	def _generate_vocab(self):
		""" represent the vocabulary as all possible two letter combinations 
			(could add numbers to make vocab larger or extend to 3 letters)
		"""
		if self.ntokens < 27:
			return self.alphabet

		vocab = []
		i, j = 0, 0
		while len(vocab) < self.ntokens:
			vocab.append(self.alphabet[i] + self.alphabet[j])
			i, j = i if j < 25 else i+1, (j + 1) % 26

		return vocab


	def _generate_tree(self, seed=1111):
		""" generate a random tree from the root with maximum degree
			of ntokens+1 for each node. associate with each edge (u,v):
			- size of the subtree starting from and including v (to compute probability of each sentence)
			- token associated with the edge (to generate sentences from the tree) 
		"""

		# set seed
		random.seed(seed)
		np.random.seed(seed)

		# initialize root
		T, nnodes = DiGraph(), 1
		T.add_node(0, depth=0, seq=[''])
		q = deque([0])

		# bfs extending
		while len(q) > 0:

			node = q.popleft()

			# if depth is maximal, continue
			node_id, node_info = T.nodes(1)[node]
			if node_info['depth'] >= self.depth:
				continue

			# with probability pstop, continue
			if random.random() < self.pstop:
				continue
			
			# determine number of edges
			if self.mode == 'uniform':
				ndesc = random.randint(1, self.ntokens)
			elif self.mode == 'poisson':
				ndesc = np.random.poisson(lam=self.lam)
				while ndesc > self.ntokens or ndesc == 0: ndesc = np.random.poisson(lam=self.lam)
			elif self.mode == 'power':
				ndesc = self._sample_power()
			else:
				raise ValueError("only uniform, poisson, power accepted atm!")

			# determine tokens
			tokens = [self.vocab[i] for i in random.sample(range(self.ntokens), ndesc)]

			# add nodes to the graph and update queue
			for token in tokens:
				T.add_node(nnodes, depth=node_info['depth']+1, seq=node_info['seq'] + [token])
				T.add_edge(node_id, nnodes)
				q.append(nnodes)
				nnodes += 1
		
		self.nnodes = nnodes
		return T

	def _collect_sentences(self):
		""" iterate over all nodes, each representing one sentence """
		
		text = []
		for node_id, node_info in self.T.nodes(1):

			# leave out the root
			if node_id == 0:
				continue

			text.append(" ".join(node_info['seq'][1:]))

		text.sort(key = lambda s: len(s))

		# atm train, test, and eval are the same
		data = dict()
		data['train'] = text
		data['test'] = text
		data['valid'] = text
		return data

	def _perplexity(self):
		''' should give lower bound for the model perplexities, we'll see if true
		'''

		# lookup holds probability for each node to get there from ancestor
		lookup = dict()
		for node_id, node_info in self.T.nodes(1):

			if node_id == 0:
				continue

			pred = self.T.predecessors(node_id)[0]
			add = (0 if pred == 0 else 1)
			ndesc = 1+len(descendants(self.T, node_id))
			prob = ndesc / (add + len(descendants(self.T, pred)))
			lookup[node_id] = prob

		# iterate over all paths from root to any node, sum log probs
		entropy = 0
		overall = 0
		for node_id, node_info in self.T.nodes(1):

			if node_id == 0:
				continue

			path = shortest_path(self.T, 0, node_id)
			local_entropy = sum([-lookup[n] * np.log(lookup[n]) for n in path[1:]])

			print([lookup[n] for n in path[1:]])

			ndesc = len(descendants(self.T, node_id))
			if  ndesc > 0:
				# this is not a leaf, add escape prob
				local_entropy += - (1 / (ndesc + 1)) * np.log(1 / (ndesc + 1))

			entropy += local_entropy
			overall += (len(path)) 

		entropy = entropy# / overall
		# ppl is exp of entropy
		entropy = entropy / overall
		ppl = math.exp(entropy)
		return ppl

	def _sample_power(self):
		p = np.power(1./np.arange(1,self.ntokens, 1), self.k)
		p = p / np.sum(p)
		p = np.cumsum(p)
		return 1 + np.searchsorted(p, random.random())



def main(argv):

	try:
		opts, args = getopt.getopt(argv, "ho:d:n:m:p:s:", ["output=", "ntokens=", "depth=", "mode=", "pstop=,", "seed="])
	except:
		print('python generate_tree_data.py -o <output path> -n <ntokens> -d <depth> -m <mode> -p <pstop> -s <seed>')
		sys.exit(2)

	for opt, arg in opts:
		
		if opt == '-o':
			basepath = arg
		elif opt == '-n':
			ntokens = int(arg)
		elif opt == '-d':
			depth = int(arg)
		elif opt == '-m':
			mode = arg
		elif opt == '-p':
			pstop = float(arg)
		elif opt == '-s':
			seed = int(arg)
		else:
			print('python generate_tree_data.py -o <output path> -n <ntokens> -d <depth> -m <mode> -p <pstop> -s <seed>')
			sys.exit(2)

	#tlg = TreeLangGenerator(ntokens=ntokens, depth=depth, mode=mode, pstop=pstop)
	#tlg.generate_sentences(seed=seed)
	#tlg.save(base=basepath)
	
	#print(tlg.ppl)

	tlg = TreeLangGenerator(2,3)
	tlg.load('../data/treelang/tiny/')
	print(tlg.ppl)

if __name__ == '__main__':
	import sys, getopt
	main(sys.argv[1:])
	
