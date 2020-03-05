import sys
import Streaming
import utilities as util
import os
# python2 run_Streaming.py [file train] [file setting] [folder model] [folder data test] [prior]
# check input
if len(sys.argv) != 6:
    print"Usage: python2 run_Streaming.py [file train] [file setting] [folder model] [folder data test] [prior]"
    exit()
filetrain = sys.argv[1]
filesetting = sys.argv[2]
folder = sys.argv[3]
filetest = sys.argv[4]
fileprior = sys.argv[5]
setting = util.read_setting(filesetting)
n_tests = 1 #
folder = "%s-TPS-%s-%s-%s"%(folder,setting['sigma'],setting['batch_size'],setting['n_topics'])
print folder
if not os.path.exists(folder):
	os.makedirs(folder)
else:
	print "Folder existed"
	exit()

ft = open(filetrain,'r')
util.write_setting(folder, setting)


strm = Streaming.Streaming(fileprior, setting['alpha'], setting['n_topics'], setting['n_terms'], setting['n_infer'], setting['learning_rate'], setting['sigma'])
(wordinds1, wordcnts1, wordinds2, wordcnts2)=util.read_test(filetest, n_tests)
mini_batch =0
while True:
	mini_batch = mini_batch+1
	(wordinds, wordcnts, full) = util.read_minibatch(ft, setting['batch_size'])
	if full ==1:
		break
	print "MINI BATCH %d"%(mini_batch)
	gamma = strm.run_stream(setting['batch_size'], wordinds, wordcnts)
	(LD, ld2) = util.compute_perplex(wordinds1, wordcnts1, wordinds2, wordcnts2, strm.beta, setting['alpha'], setting['n_topics'], 
		setting['n_terms'], setting['n_infer'], n_tests)
	print LD
	util.write_model(folder, strm.beta, strm.pi, LD, ld2, mini_batch)
util.write_beta(folder, strm.beta)
ft.close()
	
