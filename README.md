# TPS
This is an implementation of Dynamic transformation of prior knowledge into Bayesian models for data streams.

Data descriptions: \n
    Training file: In the bag of words format: \n
        N index1:count1 index2:count2 ... indexN:countN \n
    Testing folder: E.g ./model/TMN.\n
    Setting file: E.g setting.txt. \n
    Prior file: E.g: prior.glove.200d.txt.\n


Requirements:\n
	python 2.7\n

Command:\n
    python run_Streaming.py [Training file] [Setting file] [Model folder] [Testing folder] [Prior file].\n
    Example: python2 run_Streaming.py train.txt setting.txt ./model/TMN . prior.glove.200d.txt.\n
