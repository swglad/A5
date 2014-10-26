#!/usr/bin/env python
# Authors: Alex Gerstein, Scott Gladstone
# CS 73 Assignment 5: Vector Space Models
# 
from __future__ import division
import numpy
import numpy.linalg
from collections import defaultdict

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
        
    def train(self, traindata, trainlabels, max_epochs):

        #TODO: fill in
        
        return mistakes

    def test(self, testdata, testlabels):

        #TODO: fill in
        
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
    print "Finished training, with", trainmistakes/numpy.size(trainlabels), "% error rate"
    devmistakes = classifier.test(devdata, devlabels)
    print devmistakes/numpy.size(devlabels), "% error rate on development data"
    testmistakes = classifier.test(testdata, testlabels)
    print testmistakes/numpy.size(testlabels), "% error rate on test data"
