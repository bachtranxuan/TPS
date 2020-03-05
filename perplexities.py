import numpy as np
import utilities as ul
from scipy.special import digamma
def dirichlet_expectation(alpha):
    if len(alpha.shape) == 1:
        return (digamma(alpha)-digamma(sum(alpha)))
    return (digamma(alpha)-digamma(np.sum(alpha, axis=1))[:,np.newaxis])
class Stream:
    def __init__(self, beta, alpha, k, n_terms, n_infer):
        self.alpha = alpha
        self.K = k
        self.n_infer = n_infer
        self.n_terms =n_terms
        self.beta = beta
    def doc_e_step(self, batchsize, wordsind, wordscnt):
        gamma = np.random.gamma(100., 1./100., (batchsize, self.K))
        for d in range(batchsize):
            indx = wordsind[d]
            cntx = wordscnt[d]
            gammad = np.ones(self.K)*self.alpha+float(np.sum(cntx))/self.K
            expElogthetad = np.exp(ul.dirichlet_expectation(gammad))
            betad = self.beta[:,indx]
            for i in range(self.n_infer):
                phi = expElogthetad*betad.transpose()
                phi = phi/np.sum(phi, axis=1)[:,np.newaxis]
                gammad = self.alpha + np.dot(cntx, phi)
                expElogthetad = np.exp(ul.dirichlet_expectation(gammad))
            gamma[d]=gammad
            gamma[d]=gammad/np.sum(gammad)
        return (gamma)
    def compute_doc(self, gammad, wordind, wordcnt):
        ld2=0.0
        frequen=np.sum(wordcnt)
        for i in range(len(wordind)):
            p=np.dot(gammad, self.beta[:,wordind[i]])
            ld2+=wordcnt[i]*np.log(p)
        if frequen == 0:
            return ld2
        else:
            return ld2/frequen
    def compute_perplex(self, wordind1, wordcnt1, wordind2, wordcnt2):
        batchsize = len(wordind1)
        gamma = self.doc_e_step(batchsize, wordind1, wordcnt1)
        LD2=0
        for i in range(batchsize):
            LD2=LD2+self.compute_doc(gamma[i], wordind2[i], wordcnt2[i])
        return LD2/batchsize
