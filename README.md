

# TPS
This is an implementation of Dynamic transformation of prior knowledge into Bayesian models for data streams.

Some contribution of our framework
*	We propose a novel framework called Dynamic transformation of prior knowledge into Bayesian models for data streams (TPS) which is a streaming learning framework that enables the incorporation of the prior knowledge of different forms into a base Bayesian model for data streams.
*	 We show that SVB ([Broderick et al., 2013](https://arxiv.org/pdf/1307.6769.pdf)) can forget any knowledge at a rate of ![formula](https://render.githubusercontent.com/render/math?math=O(T^{-1})), after learning from more T mini-batches of data.
*	We have an extensive evaluation of different methods.

## Installation
1. Clone the repository
```
		https://github.com/bachtranxuan/TPS.git
``` 
2. Requirements environment
```
		Python 2.7
		Numpy, Scipy
```
## Training
You can run with command
```
	python run_Streaming.py [Training file] [Setting file] [Model folder] [Testing folder] [Prior file].
```
Example
```
python run_Streaming.py data/train.txt data/setting.txt data/result data data/prior.glove.200d.txt.
```
## Data descriptions
*	Training file, we used the bag of words format. (E.g data/train.txt)
```
		4 14:1 12:2 7:2 96:1
```
*	Testing folder, including one or more pair file (part_1, part_2). Each document in the test set is divided randomly into two disjoint part ![formula](https://render.githubusercontent.com/render/math?math=w_{obs}) (part_1) and ![formula](https://render.githubusercontent.com/render/math?math=w_{ho}) (part_2) with a ratio of 4:1. (E.g data).
*	Setting file contains the value of parameters (hyperparameter) of the model such as alpha (hyperparameter of the model), n_topics (number of the topic), learning_rate (learning rate in gradient descent algorithm) .... (E.g data/setting.txt). 
```
		alpha 0.01
		n_topics 50
		n_terms 2823
		batch_size 500
		n_infer 50
		learning_rate 0.01
		sigma 0.01
```
*	Prior file, includes V lines which V is the number of word of set vocab. Each line is a representation of word such as w2v which was pre-trained from 6 billion tokens of Wikipedia2014 and Gigaword5 ([JeffreyPennington., 2014](https://nlp.stanford.edu/projects/glove/)) (E.g: data/prior.glove.200d.txt).

## Result
We compare our model with four state-of-the-art base-lines:
SVB ([Broderick et al., 2013](https://arxiv.org/pdf/1307.6769.pdf)), PVB ([McInerney et al.,  2015](https://arxiv.org/pdf/1507.05253.pdf)), SVP-PP ([Masegosa et al., 2017](http://proceedings.mlr.press/v70/masegosa17a/masegosa17a.pdf)) and KPS ([Duc, Anh Nguyen et al., 2017](https://link.springer.com/chapter/10.1007/978-3-319-57529-2_20)).Log predictive probability ([LPP](http://jmlr.org/papers/v14/hoffman13a.html))  and Normalized pointwise mutual information ([NPMI](https://www.aclweb.org/anthology/E14-1056/))

![Log predictive probability](./figures/perplexities.png)
![Normalized pointwise mutual information](./figures/npmi.png)

Some topics are learned from data
```
| Topic\word  |   
|:-----------:|------------|------------|------------|------------|------------|------------|-------------|------------|------------|------------|
| Military    |    war     |    army    |   naval    |    navy    |  commader  |   commad   |   military  |   forces   |    air     |  ship     |
| Music       |  music     |   musical  |   piano    |   songs    |  composer  | orchestral | instruments |  orchestra |   vocal    |  sound    |
``` 
## Citation
if you find that TPS is useful for your research, please citing:
```
cite
```

