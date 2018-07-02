# NER

# basemodel

code for file handling / preprocessing from Nils Reimers, https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf

# experiments

Gazetteer to create distantly supervised training data has been optimized for recall, 
incorporating 1000 concepts per entity type from concept net.

## 1: estimate noise statistics on manually labelled corpus

- estimate noise statistics with confusion matrix (manually and automatically labelled on same data)
- train base model and use confusion matrix as label flip probabilities in linear layer (untrainable)
- unplugg noise model and use base model on test data

## 2: linear noise model with trace regularization

- noise model of the form: Linear Layer (Identity, trainable)
- objective function incorporates <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda&space;*&space;tr(\Psi)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda&space;*&space;tr(\Psi)" title="\lambda * tr(\Psi)" /></a>
- Linear Layer weights have to be normalized to be a stochastic matrix in every iteration
- train base and noise model simultaneously
- unplugg noise model and use base model on test data

## 3: softmax noise model with dropout regularization

- noise model of the form: Dropout (.1) -> Linear Layer (Identity, trainable) -> Softmax
- train base and noise model simultaneously
- unplugg noise model and use base model on test data

