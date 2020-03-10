

# TPS
This is an implementation of Dynamic transformation of prior knowledge into Bayesian models for data streams. Our framework enables to incorporate the prior knowledge of different forms into a base Bayesian model for data streams to improve quality and performance
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
*	Training file, we used the bag of words format.
*	Testing folder (E.g data).
*	Setting file (E.g data/setting.txt). 
*	Prior file, includes V lines which V is the number of word of set vocab. Each line is a representation of word such as w2v (E.g: data/prior.glove.200d.txt).
## Model
Graphical representation for TPS is illustrated below
\
<img src="figures/model.png" alt="centered image" height="450px" width="500px" >
## Result
We compare our model with four state-of-the-art base-lines:
SVB ([Broderick et al., 2013](https://arxiv.org/pdf/1307.6769.pdf)), PVB ([McInerney et al.,  2015](https://arxiv.org/pdf/1507.05253.pdf)), SVP-PP ([Masegosa et al., 2017](http://proceedings.mlr.press/v70/masegosa17a/masegosa17a.pdf)) and KPS ([Duc, Anh Nguyen et al., 2017](https://link.springer.com/chapter/10.1007/978-3-319-57529-2_20)).Log predictive probability ([LPP](http://jmlr.org/papers/v14/hoffman13a.html))  and Normalized pointwise mutual information ([NPMI](https://www.aclweb.org/anthology/E14-1056/))

![Log predictive probability](./figures/perplexities.png)
![Normalized pointwise mutual information](./figures/npmi.png)

Some topics are learned from data
```
Military: war army naval navy commander command military forces air ship.
Music: music musical piano songs composer orchestral instruments orchestra vocal sound.
``` 
