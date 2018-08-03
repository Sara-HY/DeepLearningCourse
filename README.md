# DeepLearningCourse
Peking University Deep Learning Course Project in TensorFlow 

## `` AutoEncoder `` 

AutoEncoder & FC Neural Network

* autoencoder.py

	A simple 2-layer auto-encoder model on mnist dataset.

* neural_network.py

	A 5-layer FC neural network on mnist dataset.

## `` CNN_OxFlowers `` 

CNN model - A 17-category Classifier for OxFlowers

* data_process.py

	Split the dataset to train, valid and test sets, and add a data-augmentation set.

* model.py

	Build a model of 3 convolution neural network layers and 2 full-connected layers.

* main.py

	Train the model with different parameters and generate the learning curve.

## ``SentimentAnalysis``

LSTM model - A 5-category Classifier for Sentiment Analysis

* utils.py

	Split the dataset to train, valid and test sets, and get the label of each other. (Please add glove.6B.300d.txt in dataset youself.)

* model.py

	Build a 3 LSTM neural network layers and 1 full-connected layer.

* main.py

	Train the model with different parameters and generate the learning curve.