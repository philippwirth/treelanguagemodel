import numpy as np
import matplotlib.pyplot as plt

def dump_contexts(contexts, basepath='', epoch=0, hsz=2, bsz=1):

	""" format: 
		- one row per line {depth1, depth2, x1, y1, x2, y2 }
	"""
	
	# the overall number of lines is sum(bsz*(seq_len - 1)) over all contexts
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
		for i in range(0, np.size(np_ctxts, 0)-1, seq_len):
			for j in range(i, i + seq_len-1, 1):

				data[n, 0] = int(j-i)				# depth of first node
				data[n, 1] = int(j-i+1) 			# depth of second node
				data[n, 2:2+hsz] = np_ctxts[j,:]	# coordinates of first node
				data[n, 2+hsz:] = np_ctxts[j+1,:]	# coordinates of second node

				n += 1

	savepath = basepath + str(epoch) + '.out'
	header = "depth 1, depth 2, context1, context2"
	formatstr = ' '.join(['%i']*2 + ['%1.3e']*2*hsz)
	np.savetxt(savepath, data, header=header, delimiter=' ', fmt=formatstr)

def load_contexts(path):
	data = np.loadtxt(path)
	return data

def visualize_contexts(data):

	maxdepth = 2
	hsz = (np.size(data, 1) - 2) // 2
	for i in range(np.size(data, 0)):

		#plt.plot([data[i,2], data[i,4]], [data[i,3], data[i,5]], 'k')
		#plt.plot(data[i,2], data[i,3], marker='o', markeredgecolor='k', markerfacecolor='k') #plt.cm.Purples(float(data[i,0])/maxdepth))
		#plt.plot(data[i,4], data[i,5], marker='o', markeredgecolor='k', markerfacecolor=plt.cm.Purples(float(data[i,1])/maxdepth))
		plt.arrow(data[i,2], data[i,3], data[i,4] - data[i,2], data[i,5] - data[i,3], width=0.0000001, head_width=0.00005, head_length=0.00005, fc='k', ec='k')


		#plt.axis('equal')
		plt.axis([-1.001, -0.999, -1.001, -0.998])


# test
if __name__ == '__main__':

	'''
	
	j = 1
	for i in range(1, 11, 1):

		plt.subplot(2, 5, j)
		plt.title("epoch: " + str(i))
		path = "../results/crossentropy_test/context_dump_" + str(i) + ".out"
		data = load_contexts(path=path)
		visualize_contexts(data)
		j += 1
	'''
	

	path = "../results/crossentropy_test/context_dump_10.out"
	data = load_contexts(path=path)
	visualize_contexts(data)
	
	plt.show()

	
	

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
	'''
	path = "../results/crossentropy_test/context_dump_10.out"
	data = load_contexts(path=path)

	i = 17
	k = 1
	while k <= 16:

		plt.subplot(4,4,k)

		visualize_contexts(data[i:i+1, :])
		k = k + 1
		i = i + 1
		print(i)
	#print(data.shape)
	plt.show()
	'''
	
	

