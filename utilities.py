import numpy as np
from scipy.special import digamma
import perplexities as per
def get_top(beta, nword=10):
	(k,words) = beta.shape
	inds = range(words)
	top=np.zeros((k,nword))
	for i in range(k):
		inds.sort(lambda x,y: -cmp(beta[i,x], beta[i,y]))
		for j in range(nword):
			top[i,j] = inds[j]
	return top
def dirichlet_expectation(alpha):
	if len(alpha.shape) == 1:
		return (digamma(alpha)-digamma(sum(alpha)))
	return (digamma(alpha)-digamma(np.sum(alpha, axis=1))[:, np.newaxis])
def read_minibatch(path, batchsize):
	wordsindex = []
	wordscount = []
	full = 0
	for i in range(batchsize):
		temp = path.readline()
		if len(temp) < 1:
			full = 1
			break
		temp = temp.split(" ")
		n_words = int(temp[0])
		ind = [0 for i in range(n_words)]
		cnt = [0 for i in range(n_words)]
		for j in range(n_words):
			term_word = temp[j+1].split(":")
			ind[j] = int(term_word[0])
			cnt[j] = int(term_word[1])
		wordsindex.append(ind)
		wordscount.append(cnt)
	return (wordsindex, wordscount, full)
def read_data(path):
	fp=open(path, 'r')
	wordsindex = []
	wordscount = []
	while True:
		temp = fp.readline().replace("\n","")
		if len(temp) < 1:
			break
		temp = temp.split(" ")
		n_words = int(temp[0])
		ind = [0 for i in range(n_words)]
		cnt = [0 for i in range(n_words)]
		for j in range(n_words):
			term_word = temp[j+1].split(":")
			ind[j] = int(term_word[0])
			cnt[j] = int(term_word[1])
		wordsindex.append(ind)
		wordscount.append(cnt)
	fp.close()
	return (wordsindex, wordscount)
def read_test(path, n_tests):
	corpus_wordinds1=[]
	corpus_wordcnts1=[]
	corpus_wordinds2=[]
	corpus_wordcnts2=[]
	for i in range(n_tests):
		part1 ='%s/data_test_%d_part_1.txt'%(path,i+1)
		part2 ='%s/data_test_%d_part_2.txt'%(path,i+1)
		(x,y) = read_data(part1)
		(a,b) = read_data(part2)
		corpus_wordinds1.append(x)
		corpus_wordcnts1.append(y)
		corpus_wordinds2.append(a)
		corpus_wordcnts2.append(b)
	return(corpus_wordinds1, corpus_wordcnts1, corpus_wordinds2, corpus_wordcnts2)
	
def read_setting(path):
	filesetting = open(path, 'r')
	setting = {}
	sets = []
	vals = []
	while True:
		temp = filesetting.readline()
		if len(temp) < 1:
			break
		temp = temp.split(" ")
		sets.append(temp[0])
		vals.append(float(temp[1]))
	setting = dict(zip(sets, vals))
	setting['n_topics'] = int(setting['n_topics'])
	setting['n_terms'] = int(setting['n_terms'])
	setting['batch_size'] = int(setting['batch_size'])
	setting['n_infer'] = int(setting['n_infer'])
	filesetting.close()
	return (setting)
def read_prior(path):
	prior = []
	fp = open(path, 'r')
	while True:
		temp = fp.readline().strip()
		if len(temp) < 1:
			break
		temp = temp.split(" ")
		p=[]
		for i in range(len(temp)):
			p.append(float(temp[i]))
		p.append(1.0)
		prior.append(np.asarray(p))
	fp.close()
	return(np.asarray(prior))
def write_setting(folder, setting):
	filename ='%s/setting.txt'%(folder)
	fp=open(filename,'w')
	key = setting.keys()
	val = setting.values()
	for i in range(len(key)):
		fp.write('%s %f'%(key[i],val[i]))
	fp.close()
def write_model(folder, beta, pi, LD, ld2, minibatch):
	per = "%s/perplexities.csv"%(folder)
	fp = open(per,'a')
	fp.write(' %f,'%(LD))
	fp.close()

	per = "%s/perplexities_pairs.csv"%(folder)
	fp = open(per,'a')
	for ld in ld2:
		fp.write(' %f,'%(ld))
	fp.write('\n')
	fp.close()
	
	bt = "%s/top20_1_%d.dat"%(folder, minibatch)
	fp = open(bt,'w')
	top = get_top(beta, 20)
	for k in range(len(top)):
		fp.writelines('%d '% temp for temp in top[k])
		fp.write('\n')
	fp.close()
def write_beta(folder, beta):
	path = "%s/beta_final.dat"%(folder)
	fp = open(path,'w')
	for k in range(len(beta)):
		fp.writelines('%f '% temp for temp in beta[k])
		fp.write('\n')
	fp.close()
def compute_perplex(wordinds1, wordcnts1, wordinds2, wordcnts2, beta, alpha, k, n_terms, n_infer, n_tests):
	pr = per.Stream(beta, alpha, k, n_terms, n_infer)
	LD=0
	ld2 = []
	for k in range(n_tests):
		ld=pr.compute_perplex(wordinds1[k], wordcnts1[k], wordinds2[k], wordcnts2[k])
		LD=LD+ld
		ld2.append(ld)
	LD=LD/n_tests
	return (LD, ld2)
def compute_beta(prior, pi):
	beta= np.dot(pi, prior.transpose())
	beta = np.exp(beta)
	beta = beta/np.sum(beta,axis=1)[:,np.newaxis]
	return beta

