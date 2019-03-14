import numpy as np
import matplotlib.pyplot as plt

def dump_contexts(contexts, basepath='', epoch=0, hsz=2, bsz=1):

	""" format: 
		- one row per line {depth1, depth2, x1, y1, x2, y2 }
	"""
	
	# the overall number of lines is sum(bsz*(seq_len)) over all contexts
	nlines = sum([len(ctxts)-bsz for ctxts in contexts])
	data = np.zeros((nlines, 2*hsz + 2))
	
	n = 0
	for ctxts in contexts:

		# convert tensor to numpy
		np_ctxts = ctxts.detach().cpu().numpy()
		#np_ctxts = ctxts

		# every batch is bsz*seq_len rows
		seq_len = np.size(np_ctxts, 0) // bsz

		# iterate over all sequences and extract plottable lines
		for i in range(0, np.size(np_ctxts, 0), seq_len):
			for j in range(i, i + seq_len-1, 1):

				data[n, 0] = int(j-i + 0)			# depth of first node
				data[n, 1] = int(j-i+ 1 ) 			# depth of second node
				data[n, 2:2+hsz] = np_ctxts[j,:]	# coordinates of first node
				data[n, 2+hsz:] = np_ctxts[j+1,:]	# coordinates of second node

				n += 1

	savepath = basepath + str(epoch) + '.out'
	header = "depth 1, depth 2, context1, context2"
	formatstr = ' '.join(['%i']*2 + ['%1.8e']*2*hsz)
	np.savetxt(savepath, data, header=header, delimiter=' ', fmt=formatstr)

def load_contexts(path):
	data = np.loadtxt(path)
	return data

def visualize_contexts(data, title=''):

	maxdepth = 4
	hsz = (np.size(data, 1) - 2) // 2
	for i in range(0, np.size(data, 0)):

		plt.plot([data[i,2], data[i,4]], [data[i,3], data[i,5]], 'k')
		c1 = plt.cm.Greys(float(data[i,0])/maxdepth)
		c2 = plt.cm.Greys(float(data[i,1])/maxdepth)
		plt.plot(data[i,2], data[i,3], marker='o', markeredgecolor='k', markerfacecolor=c1) #plt.cm.Purples(float(data[i,0])/maxdepth))
		plt.plot(data[i,4], data[i,5], marker='o', markerfacecolor=c2, markeredgecolor='k')
		#plt.arrow(data[i,2], data[i,3], data[i,4] - data[i,2], data[i,5] - data[i,3], width=0.0001, head_width=0.0001, head_length=0.0001, fc='k', ec='k')


		plt.title(title)
		plt.axis('equal')
		#plt.xlim(-0.01, 0.05)
		#plt.ylim(-0.005, 0.03)
		#plt.axis([-0.005, 0.01, -0.005, 0.05])

# test
if __name__ == '__main__':

	
	'''
	j = 1
	for i in range(1, 51, 5):

		plt.subplot(2, 5, j)
		plt.title("epoch: " + str(i))
		path = "../results/test/context_dump_" + str(i) + ".out"
		data = load_contexts(path=path)
		visualize_contexts(data)
		j += 1
	
	plt.show()
	'''
	
	'''
	path = "../results/test/context_dump_50.out"
	data = load_contexts(path=path)
	visualize_contexts(data)
	
	plt.show()
	'''


	
	

	'''
	
	path = "../results/crossentropy_test/context_dump_10.out"
	data = load_contexts(path=path)

	i = 0
	k = 1
	while k <= 12:

		plt.subplot(3,4,k)

		j = i+1
		while (j < 33 and data[j,0] > 0):
			j += 1

		print([i,j])
		visualize_contexts(data[i:j, :])
		k = k + 1

		i = j
	#print(data.shape)
	plt.show()
	'''
	
	base = []
	da, db = [], []
	for i in range(50):

		path = "../results/test/context_dump_" + str(i) + ".out"
		data = load_contexts(path=path)

		x0 = np.zeros(2)
		x1 = data[0,4:6]
		x2 = data[1,4:6]

		base.append(i)
		print([x0, x1])
		print([x0, x2])
		da.append(np.exp(-10*pow(np.linalg.norm(x1-x0),1)))
		db.append(np.exp(-10*pow(np.linalg.norm(x2-x0),1)))


	plt.plot(base, [a / (a + b) for (a,b) in zip(da, db)], 'r')
	plt.plot(base, [b / (a + b) for (a,b) in zip(da, db)], 'g')
	plt.show()

	
	
	'''

	#i = 8
	epoch = 10
	path = "../results/merity_tiny_gru/context_dump_" + str(epoch) + ".out"
	data = load_contexts(path=path)

	i = 17
	k = 1
	titles = ['A A B A', 'A A A A', 'A A A B', 'B A A B']
	while k <= 4:

		visualize_contexts(data[i:i+4, :], title="epoch: "+str(epoch))#, title=titles[k-1])
		print(data[i:i+4])
		k = k + 1
		i = i + 4
	
	#plt.axis('equal')
	plt.show()
	'''
		

	
	
	

