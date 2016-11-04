# Mushroom-Classifier
Contains a neural network powered binary-classifier for the [UC Irvine Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom).

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Dependencies](#dependencies)
- [Usage Instructions](#usage-instructions)
  - [Install dependencies](#install-dependencies)
  - [Get data set and generate feature vectors](#get-data-set-and-generate-feature-vectors)
  - [Data Set splitting](#data-set-splitting)
  - [Run and test model](#run-and-test-model)
- [Data Pre Processing](#data-pre-processing)
- [Neural Network Specifications](#neural-network-specifications)
- [Classifier Performance](#classifier-performance)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
## Dependencies

- Python 2.7
- [tensorflow](https://github.com/tensorflow/tensorflow/)
- [tflearn](https://github.com/tflearn/tflearn)
- [hickle](https://github.com/telegraphic/hickle)

## Usage Instructions
### Install dependencies
Run ```source setup.sh``` or ```./setup.sh```  
In case you run into issues with permissions, run ```sudo chmod +x setup.sh``` and try running the file again.

### Get data set and generate feature vectors
Run ```generate_datasets.py```
### Data Set splitting
This stage involves dividing the data set into three parts:
  
- Training
- Validation
- Testing  

By default, they are split in the ratio 9:1:1 (Training: Validation : Testing).
This ratio can be modified by changing the ```TRAIN```, ```VALID``` and ```TEST``` constants in ```split_datasets.py```
When ready, run ```split_datasets.py```

### Run and test model
run ```nn_model.py```

## Data Pre Processing
For each feature, attributes are [one-hot encoded](https://en.wikipedia.org/wiki/One-hot).
Missing values are represented as an independent bit in the one-hot encoded representation.
These encoded attributes are then chained together to form a 126-bit long feature vector. 

## Neural Network Specifications
This binary classifier uses one hidden layer in addition to an input and output layer.  
The particulars of each layer are described as under:

- Input Layer (126 nodes)
- Hidden Layer (64 nodes; activation function : [relu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)))
- Output Layer (2 nodes; activation function : [softmax](https://en.wikipedia.org/wiki/Softmax_function))

## Classifier Performance
On training the classifier on 90 % of the data (80% training + 10% validation), this model has achieved an accuracy of 100% on unseen Test data! Yay!
