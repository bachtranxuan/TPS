import numpy as np
import time
import utilities as ul
class Streaming:
	def gradient(self, pit,beta, wordsinds, wordscnts, expElogtheta):
		#derivative pi_k
		#SUM_d^D SUM_n^N_d SUM_v^V Phi_dnk [I_w_dn  = v ] * \\
		#[eta_v - (SUM_i^V eta_i * exp(pi_K, eta_i))/(SUM_j^V exp(pi_k, eta_j))]
		sumphi=np.zeros(self.K)
		batchsize = len(wordsinds)

		# Compute (SUM_i^V eta_i * exp(pi_K, eta_i))/(SUM_j^V exp(pi_k, eta_j))
		dl = np.zeros((self.K, self.length))
		temp = np.dot(pit, self.prior_knowledge.transpose())
		temp = np.exp(temp)
		sumtemp = np.sum(temp, axis=1)
		for n in range(self.n_terms):
			dl+=np.outer(temp[:,n],self.prior_knowledge[n])
		dl/=sumtemp[:,np.newaxis]

		delta = np.zeros((self.K, self.length))
		for d in range(batchsize):
			expElogthetad = expElogtheta[d]
			indx = wordsinds[d]
			cntx = wordscnts[d]
			betad = beta[:,indx]
			phi = expElogthetad * betad.transpose()
			phi = phi / np.sum(phi, axis=1)[:, np.newaxis]
			for n in range(len(indx)):
				x = cntx[n] * phi[n]
				sumphi += x
				delta += np.outer(x, self.prior_knowledge[indx[n]])
		S =(delta-(sumphi*dl.T).T-self.sigma*(pit-self.pi))/batchsize
		S *=self.learning_rate
		pit+=S
		return (pit)
	def __init__(self, path_prior, alpha, k, n_terms, n_infer, learning_rate, sigma):
		# Implements VB for Dynamic transformation of prior knowledge into Bayesian models for data streams
		"""
		:param path_prior: Path file prior
		:param alpha: Hyperparameter for theta
		:param k: Number of topics
		:param n_terms: Number of words in vocab
		:param n_infer: Number of iterate between gamma and phi until convergence
		:param learning_rate: Parameter of gradient ascent method [0.01]
		:param sigma: Parameter of model
		"""
		self.prior_knowledge = ul.read_prior(path_prior)
		self.alpha = alpha
		self.K = k
		self.n_infer = n_infer
		self.n_terms =n_terms
		self.learning_rate = learning_rate
		self.sigma =sigma
		self.length = len(self.prior_knowledge[0])
		# print "Num dimension of prior knowledge: %d"%(self.length)
		self.pi = np.random.rand(self.K, self.length)
		self.pi = self.pi/np.sum(self.pi,axis=1)[:,np.newaxis]
		self.beta = ul.compute_beta(self.prior_knowledge, self.pi)
	def doc_e_step(self, batchsize, wordsinds, wordscnts):

		gamma = np.random.gamma(100., 1./100., (batchsize, self.K))
		Elogtheta = ul.dirichlet_expectation(gamma)
		expElogtheta = np.exp(Elogtheta)
		beta = self.beta
		for d in range(batchsize):
			indx = wordsinds[d]
			cntx = wordscnts[d]
			# Initialize the variational distribution q(theta|gamma) for each mini-batch
			gammad = np.ones(self.K)*self.alpha+float(np.sum(cntx))/self.K
			expElogtheta[d] = np.exp(ul.dirichlet_expectation(gammad))
			betad = beta[:,indx]
			for i in range(self.n_infer):
				# We update local parameter phi and gamma as in Latent Dirichlet Allocation [Blei et al. (2003).]
				phi = expElogtheta[d]*betad.transpose()
				phi /=np.sum(phi, axis=1)[:,np.newaxis]
				gammad = self.alpha + np.dot(cntx, phi)
				expElogtheta[d] = np.exp(ul.dirichlet_expectation(gammad))
	    		gamma[d]=gammad
		return (gamma, expElogtheta)
	def update(self, expElogtheta, wordinds, wordcnts):
		pit = np.copy(self.pi)
		start = time.time()
		for i in range(75):
			pit = self.gradient(pit,self.beta, wordinds, wordcnts, expElogtheta)
		end = time.time()
		print "Time update pi: %s"%(end-start)

		self.pi = pit
		self.beta = ul.compute_beta(self.prior_knowledge, self.pi)
	def run_stream(self, batchsize, wordinds, wordcnts):
		(gamma, expElogtheta) = self.doc_e_step(batchsize, wordinds, wordcnts)
		self.update(expElogtheta, wordinds, wordcnts)
		return (gamma)

