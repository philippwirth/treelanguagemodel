
class TreeLanguageGenerator:

	def __init__(self, ntokens, depth):

		if ntokens > 676: 	   # vocab not representable by two letter combinations (TODO)
			raise ValueError("ntokens > 676 (not representable by two letters)")

		self.ntokens = ntokens # vocabulary size
		self.depth = depth     # maximum length for sentence

		self.vocab = None	   # vocabulary
		self.data = None	   # data (text corpus)
		self.T = None		   # language tree

		self.perplexity = 0	   # (OPTIONAL) compute perplexity of tree
		self.alph = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
					 'n', 'o', 'p','q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

	def generate_sentences(self, seed=1111):
		
		# generate the vocabulary
		self.vocab = self._generate_vocab()

		# generate the tree
		self.T = self._generate_tree(seed)

		# traverse the tree to collect sentences
		self.data = self._collect_sentences()


	def save(self):
		pass

	def load(self):
		pass

	def _generate_vocab(self):
		""" represent the vocabulary as all possible two letter combinations 
			(could add numbers to make vocab larger or extend to 3 letters)
		"""
		
		vocab = []
		i, j = 0, 0
		while len(vocab) < self.ntokens:
			vocab.append(self.alph[i] + self.alph[j])
			i, j = i if j < 25 else i+1, (j + 1) % 26

		return vocab


	def _generate_tree(self, seed=1111):
		""" generate a random tree from the root with maximum degree
			of ntokens+1 for each node. associate with each edge (u,v):
			- size of the subtree starting from and including v (to compute probability of each sentence)
			- token associated with the edge (to generate sentences from the tree) 
		"""
		pass

	def _collect_sentences(self):
		pass

	def _perplexity(self):
		return 0


if __name__ == '__main__':
	tlg = TreeLanguageGenerator(ntokens=10, depth=5)
	tlg.generate_sentences()
	print(tlg.vocab)
