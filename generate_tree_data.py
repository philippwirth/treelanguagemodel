
class TreeLanguageGenerator:

	def __init__(self, ntokens, depth):

		self.ntokens = ntokesn # vocabulary size
		self.depth = depth     # maximum length for sentence

		self.vocab = None	   # vocabulary
		self.data = None	   # data (text corpus)
		self.T = None		   # language tree

		self.perplexity = 0	   # (OPTIONAL) compute perplexity of tree

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
		pass

	def _generate_tree(self, seed=1111):
		pass

	def _collect_sentences(self):
		pass

	def _perplexity(self):
		return 0