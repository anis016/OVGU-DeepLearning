# Introduction to Deep Learning

### Assignment 1
Implement and train a deep model on a MNIST image classification task.

Details: [Assignment 1](https://ovgu-ailab.github.io/idl2018/ass1.html)

#### Preparing the MNIST dataset
1. Download the raw MNIST files from [`Yann LeCun’s website`](http://yann.lecun.com/exdb/mnist/)
2. Unzip the data files.
3. Copy the files in to `course_intro-to-dl/data/MNIST`
4. Download the [`coversion.py`](https://ovgu-ailab.github.io/idl2018/assignments/1/conversions.py) script.
5. Run `python conversions.py -c -n`

#### Model
Multi-layer perceptron

#### Accuracy
Accuracy: 89.2188%

### Assignment 2
Basic Visualization on a MNIST image classification task.

Details: [Assignment 2](https://ovgu-ailab.github.io/idl2018/ass2.html)

#### Preparing the MNIST dataset
1. Follow this to [`import data into google colaboratory`](https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory#answer-47019779)

#### Setup Tensorboard

2. Setup [`tensorboard in google colab`](https://www.dlology.com/blog/quick-guide-to-run-tensorboard-in-google-colab/).

### Assignment 3
Create a model for the MNIST dataset using convolutional neural networks (CNN).

Details: [Assignment 3](https://ovgu-ailab.github.io/idl2018/ass3.html)

#### Preparing the MNIST and Fashion-MNIST dataset
1. Download the raw MNIST files from [`Yann LeCun’s website`](http://yann.lecun.com/exdb/mnist/)
2. Download the raw Fashion-MNIST files from [`Fashion-MNIST`](https://github.com/zalandoresearch/fashion-mnist#loading-data-with-other-machine-learning-libraries)
3. Unzip the data files.
4. Copy the files in to `course_intro-to-dl/data/MNIST`
5. Download the [`coversion.py`](https://ovgu-ailab.github.io/idl2018/assignments/1/conversions.py) script.
6. Run `python conversions.py -c -n`

#### Model
Convolutional Neural Networks

#### Accuracy
MNIST Accuracy: 97.3600%

Fashion MNIST Accuracy: 83.1400%