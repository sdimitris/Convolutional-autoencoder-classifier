# Convolutional-autoencoder-classifier
Convolutional autoencoder classifier on mnist data

In order to run this script faster you have to download the cuDNN 9.0 (Neural Network library). 
Also make sure that you have installed python 3.6 ,  the Tensorflow library and the Keras backend API.
It is important to download the MNIST data set locally.
## Need to know
You need to give the same  number of "Layers" argument when the script asks you to give it on both scripts.

Considering that you've run the autoencoder script and you've saved the autoencoder trained model , you'll have to run the classification script with the 
saved autoencoder trained model as an argument.

## Purpose of this project
The purpose of this project is to understand how the convolutional neural network works and how the train parameters ( epochs , batch size, filters of each layers, convolution layers etc)
affect the results of the classification.
## Execution instructions
#### python autoencoder.py -d "data"
#### python classification.py -d "data" -dl "data labels" -t "test data" -tl " test labels" -model "autoencoder model"
