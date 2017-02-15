"""
Test code for neural networks and deep learning

Author: Tong Qiao
Email: tong.qiao@hotmail.com
Date: 15/02/2017
"""

# Libraries
import os

# Change working directory
os.chdir('/Users/tongqiao/neural-networks-and-deep-learning/src')
import mnist_loader, network

# Load date
os.chdir('/Users/tongqiao/neural-networks-and-deep-learning')
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

"""
--------------------
Code for chapter 1
--------------------
"""
# Build a neural network with 30 hidden layers
net = network.Network([784, 30, 10])
# Train the neural network over 30 epochs, with a mini-batch size of 10, 
# and a learning rate of 3.0
net.SGD(training_data, 30, 10, 3.0, test_data = test_data)


