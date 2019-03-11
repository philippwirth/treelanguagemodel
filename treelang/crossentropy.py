import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from treelang.eucl_distance import eucl_dist_square, eucl_entailcone_dist

class TreelangCrossEntropyLoss(nn.Module):
	''' Computes cross entropy based on the treelang model: p(w|c) ~ e^-d(c, [w,c])^2
	''' 

	def __init__(self, ntokens=3, distance='eucl'):

		super(TreelangCrossEntropyLoss, self).__init__()

		self.ntokens = ntokens
		self.words = torch.LongTensor([i for i in range(self.ntokens)]).contiguous()
		self.distance = eucl_entailcone_dist if distance == 'entailcone' else eucl_dist_square

		self.loss = nn.CrossEntropyLoss()

	def forward(self, model, hiddens, targets, words=None):
		'''
			model: RNN
			hiddens: outputs of RNN for t in 1...T-1 ()
			targets: words of seq at time t in 2...T ()
		'''

		# words to cuda
		words = words.view(self.ntokens, 1).t().contiguous()
		words = words.cuda()

		# for i in range seq_len! do all this
		total_loss = 0
		seq_len = len(targets)
		for i in range(seq_len):

			# replicate h_t-1 to shape (n_layers*ndir x n_words x hsz)
			last_hidden = hiddens[i][:]
			h = last_hidden.expand(self.ntokens, -1)

			# forward pass through RNN to get output (seq_len*n_words, ndir*hsz)
			output, hidden = model(words, [h.view(1, self.ntokens, model.nhid).contiguous()])

			# compute squared distances
			d = -self.distance(last_hidden, output)

			# use CrossEntropyLoss to compute the loss and average
			# input is of size (bsz x n_words)
			total_loss += self.loss(d.view(1, self.ntokens), targets[i].view(1))

		return (total_loss / seq_len).type_as(model.decoder.weight)


if __name__ == '__main__':

	criterion = nn.CrossEntropyLoss()
	target = torch.empty(1, dtype=torch.long).random_(4)
	pdist = nn.PairwiseDistance(p=2)
	input1 = torch.randn(1, 2)
	input2 = torch.randn(4, 2)


	input1.requires_grad_(True)
	input2.requires_grad_(True)

	optimizer = torch.optim.SGD([input1, input2], lr=0.1)
	losses = []
	for i in range(500):

		optimizer.zero_grad()

		output = pdist(input1, input2)
		output = -torch.mul(output, output)
		
		loss = criterion(output.view(1,4), target)
		print(loss)
		losses.append(loss.data)

		loss.backward()

		plt.plot(input2.t().detach().numpy()[0,:], input2.t().detach().numpy()[1,:],  marker='o', linestyle='None', color=plt.cm.viridis(i / 500))
		plt.plot(input1.detach().numpy()[0][0], input1.detach().numpy()[0][1], marker='o', linestyle='None', color=plt.cm.plasma(i / 500))
		

		optimizer.step()
	print(input1)
	print(input2)
	print(target)
	plt.show()
