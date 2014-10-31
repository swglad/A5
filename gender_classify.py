#!/usr/bin/env python
# Authors: Alex Gerstein, Scott Gladstone
# CS 73 Assignment 5: Vector Space Models
# 
# Description: Perception learning model for tweet gender classification
# Parameters:
#       alpha   = 1
#       w[0]    = 1
#       ndims   = None  (dimensionality reduction)
# Enhancements: 
#       Additional features (i.e. unigrams/bigrams, avg. word or tweet 
#                                 length, number words not in dictionary,
#                                 % capitalized, % slang terms, etc.)
#       Averaged perceptron method: 1-4% error reduction
#       Change alpha with iterations (inverse: alpha ~= 1/epoch)
#       Other optimizations

from __future__ import division
import numpy
import numpy.linalg
from collections import defaultdict
from random import shuffle

def dimensionality_reduce(data, ndims):
    U, s, Vh = numpy.linalg.svd(data)
    sigmatrix = numpy.matrix(numpy.zeros((ndims, ndims)))
    for i in range(ndims):
        sigmatrix[i, i] = s[i]
    return numpy.array(U[:, 0:ndims] * sigmatrix)
        
class Perceptron:
    def __init__(self, numfeats):
        self.numfeats = numfeats
        self.w = numpy.zeros((numfeats+1,))   #+1 including intercept
        self.w[0] = 1
        self.alpha = 1  # learning rate
        # Model enhancement switches (bool)
        self.avgPerceptron = True  # average perceptron method
        self.reduceAlpha = True     # reduce alpha with epoch number
        self.shuffleData = True     # mix data order 

    # Returns updated weight vector for perceptron model
    def update(self, weight_vec, data_vec, alpha, label):
        dot = numpy.dot(weight_vec, data_vec)
        if numpy.sign(dot) == label:
            return weight_vec
        else:
            return weight_vec + (label * alpha * data_vec)
    
    # Learn perceptron hyperplane (weight vector: self.w)
    def train(self, traindata, trainlabels, max_epochs):
        
        # Add intercept feature to training data
        new_traindata = [ [0.0] * (self.numfeats+1) for i in range(len(traindata))]
        for p in range(len(traindata)):
            new_point = numpy.zeros((self.numfeats+1))
            new_point[0] = 1               # set intercept
            new_point[1:] = traindata[p]   # copy over data
            new_traindata[p] = new_point
        traindata = new_traindata

        # Learn perception if iter < max_epochs and mistakes > 0:
        avgPercepList = []
        for epoch in range(1,max_epochs):
            
            mistakes = self.test(traindata, trainlabels)
            if (mistakes == 0):
                if (self.avgPerceptron): # average weight vectors
                    self.w = sum(avgPercepList) / len(avgPercepList)
                print "Completed training in", epoch-1, "iterations."
                return mistakes
            
            data = zip(traindata, trainlabels)
            if (self.shuffleData): # mix data order
                shuffle(data) 
            for point, label in data: 
                self.w = self.update(self.w, point, self.alpha, label)
            if (self.avgPerceptron): 
                avgPercepList.append(self.w)
            if (self.reduceAlpha): # trial and error, not optimized
                self.alpha -= numpy.floor(epoch / (0.5 * max_epochs)) / 100.0

        if (self.avgPerceptron): # average weight vectors
            self.w = sum(avgPercepList) / len(avgPercepList)

        return mistakes

    # Return mistake count from inference checking: | dot(w, data) not label |
    def test(self, testdata, testlabels):
        
        # Fix feature set size if not equal to weight vector
        new_data = [ [0.0] * (self.numfeats+1) for i in range(len(testdata))]
        if len(testdata[0]) != len(self.w):
            for p in range(len(testdata)):
                new_point = numpy.zeros((self.numfeats+1))
                new_point[0] = 1              # set intercept
                new_point[1:] = testdata[p]   # copy over data
                new_data[p] = new_point
            testdata = new_data
        
        # Perform inference error computation
        mistakes = 0
        for point, label in zip(testdata, testlabels):
            mistakes += (numpy.sign(numpy.dot(self.w, point)) != label)
        
        return mistakes

def rawdata_to_vectors(filename, ndims):
    """reads raw data, maps to feature space, 
    returns a matrix of data points and labels"""
    
    spam = open(filename).readlines()
    
    labels = numpy.zeros((len(spam),), dtype = numpy.int)  #gender labels for each user
        
    contents = []
    for li, line in enumerate(spam):
        line = line.split('\t')
        contents.append(map(lambda x:x.split(), line[2:]))  #tokenized text of tweets, postags of tweets
        gender = line[1]
        if gender=='F':
            labels[li] = 1
        else:
            labels[li] = -1

    representations, numfeats = bagofwords(contents)   #TODO: change to call your feature extraction function
    print "Featurized data"

    #convert to a matrix representation
    points = numpy.zeros((len(representations), numfeats))
    for i, rep in enumerate(representations):
        for feat in rep:
            points[i, feat] = rep[feat]

        #normalize to unit length
        l2norm = numpy.linalg.norm(points[i, :])
        if l2norm>0:
            points[i, :]/=l2norm

    if ndims:
        points = dimensionality_reduce(points, ndims)
        
    print "Converted to matrix representation"

    return points, labels
        
def bagofwords(contents):
    """represents data in terms of word counts.
    returns representations of data points as a dictionary, and number of features"""
    feature_counts = defaultdict(int)  #total count of each feature, so we can ignore 1-count features
    features = {}   #mapping of features to indices
    cur_index = -1
    representations = [] #rep. of each data point in terms of feature values
    for words, postags in contents:
        for word in words:
            feature_counts[word]+=1
            
    for i, content in enumerate(contents):
        words, postags = content
        representations.append(defaultdict(float))
        for word in words:
            if word in ['<s>', '</s>'] or feature_counts[word]==1:
                continue
            if word in features:
                feat_index = features[word]
            else:
                cur_index+=1
                features[word] = cur_index
                feat_index = cur_index
            representations[i][feat_index]+=1

    return representations, cur_index+1

if __name__=='__main__':
    points, labels = rawdata_to_vectors('tweets1000.txt', ndims=None)
    #TODO: can change ndims to an integer to reduce dimensionality using SVD.
    #with None, there is no dimensionality reduction
    
    ttsplit = int(numpy.size(labels)/10)  #split into train, dev, and test 80-10-10
    traindata, devdata, testdata = numpy.split(points, [ttsplit*8, ttsplit*9])
    trainlabels, devlabels, testlabels = numpy.split(labels, [ttsplit*8, ttsplit*9])
    
    numfeats = numpy.size(traindata, axis=1)
    classifier = Perceptron(numfeats)
    
    print "Training..."
    trainmistakes = classifier.train(traindata, trainlabels, max_epochs = 30)
    print "Finished training, with", trainmistakes/numpy.size(trainlabels) * 100, "% error rate"

    devmistakes = classifier.test(devdata, devlabels)
    print devmistakes/numpy.size(devlabels) * 100, "% error rate on development data"

    testmistakes = classifier.test(testdata, testlabels)
    print testmistakes/numpy.size(testlabels) * 100, "% error rate on test data"

