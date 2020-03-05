# TPS
This is an implementation of Dynamic transformation of prior knowledge into Bayesian models for data streams.

Data descriptions: \
>	Training file: In the bag of words format: \
>>	N index1:count1 index2:count2 ... indexN:countN \
    >Testing folder: E.g ./model/TMN.\
    >Setting file: E.g setting.txt. \
    >Prior file: E.g: prior.glove.200d.txt.\

Requirements:\
	>python 2.7\

Command:\
    >python run_Streaming.py [Training file] [Setting file] [Model folder] [Testing folder] [Prior file].\
    >Example: python2 run_Streaming.py train.txt setting.txt ./model/TMN . prior.glove.200d.txt.\
