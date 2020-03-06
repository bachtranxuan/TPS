# TPS
This is an implementation of Dynamic transformation of prior knowledge into Bayesian models for data streams.

Data descriptions: 
*	Training file, we used the bag of words format.
*	Testing folder (E.g data).
*	Setting file (E.g data/setting.txt). 
*	Prior file, includes V lines which V is the number of word of set vocab. Each line is a representation of word such as w2v (E.g: data/prior.glove.200d.txt).

Requirements:
*	python 2.7

Run the demo:
*	python run_Streaming.py [Training file] [Setting file] [Model folder] [Testing folder] [Prior file].
*	Example: python2 run_Streaming.py data/train.txt data/setting.txt data/result data data/prior.glove.200d.txt.
