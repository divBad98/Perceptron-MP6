# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018

"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np

NUM_TRAINING_IMAGES = 7500
NUM_DEVELOPMENT_IMAGES = 2500
NUM_DIMENSIONS = 3072

class NearestNeighbor(object):
    def _init_(self):
        pass
    def train(self, X, y):
        self.Xtr = X
        self.ytr = y
    def predict_L1(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
            return Ypred
    def predict_L2(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        for i in range(num_test):
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
        return Ypred

def classify(train_set, train_labels, dev_set, learning_rate, max_iter):
    """
    train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
                This can be thought of as a list of 7500 vectors that are each
                3072 dimensional.  We have 3072 dimensions because there are
                each image is 32x32 and we have 3 color channels.
                So 32*32*3 = 3072
    train_labels - List of labels corresponding with images in train_set
    example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
             and X1 is a picture of a dog and X2 is a picture of an airplane.
             Then train_labels := [1,0] because X1 contains a picture of an animal
             and X2 contains no animals in the picture.

    dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
              It is the same format as train_set
    """
    perceptron = np.zeros(NUM_DIMENSIONS)
    bias = 0
    for iter_num in range(max_iter):
        for image_num in range(NUM_TRAINING_IMAGES):
            #Calculate the classification of the image based on the sign
            #of the dot product of the perceptron weights and the image dimensions
            score = np.sign(np.dot(perceptron, train_set[image_num, :]) + 1 * bias)
            if (score >= 0 and train_labels[image_num] == 0) or (score < 0 and train_labels[image_num] == 1):
                #we want the y for adjustment to be either 1 or -1
                y = 0
                if train_labels[image_num]:
                    y = 1
                else:
                    y = -1
                #incorrectly classified, so we have to adjust the weights
                adjustment = learning_rate * y * train_set[image_num, :]
                perceptron += adjustment
                bias += learning_rate * y * 1
    #Sweet, now classify the development set
    dev_labels = []
    for image_num in range(NUM_DEVELOPMENT_IMAGES):
        score = np.sign(np.dot(perceptron, dev_set[image_num, :]) + 1 * bias)
        if score >= 0:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    return dev_labels

def classifyEC(train_set, train_labels, dev_set, learning_rate, max_iter):
    # Write your code here if you would like to attempt the extra credit
    nn = NearestNeighbor()
    nn.train(train_set, train_labels)
    dev_labels = []
    for image_num in range(NUM_DEVELOPMENT_IMAGES):
        score = np.sum(np.abs(train_set[image_num,:] - dev_set[image_num,:]))
        if score >= 0:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    return dev_labels
