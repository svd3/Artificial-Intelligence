import collections
import numpy as np

############################################################
# Problem 4.1

def runKMeans(k,patches,maxIter):
    """
    Runs K-means to learn k centroids, for maxIter iterations.
    
    Args:
      k - number of centroids.
      patches - 2D numpy array of size patchSize x numPatches
      maxIter - number of iterations to run K-means for

    Returns:
      centroids - 2D numpy array of size patchSize x k
    """
    # This line starts you out with randomly initialized centroids in a matrix 
    # with patchSize rows and k columns. Each column is a centroid.
    centroids = np.random.randn(patches.shape[0],k)
    #numPatches = patches.shape[1]
    for i in range(maxIter):
        clus = [[] for i in range(k)]
        # BEGIN_YOUR_CODE (around 19 lines of code expected)
        #raise "Not yet implemented"
        # END_YOUR_CODE
        for p in range(patches.shape[1]):
            temp = []
            for j in range(k):
                temp.append(np.linalg.norm(centroids[:,j] - patches[:,p]))
            clus[temp.index(max(temp))].append(p)
        #Update centroids
        for j in range(k):
            if(len(clus[j]) > 0):
                centroids[:,j] = np.mean(patches[:,clus[j]], axis = 1)
    return centroids

############################################################
# Problem 4.2

def extractFeatures(patches,centroids):
    """
    Given patches for an image and a set of centroids, extracts and return
    the features for that image.
    
    Args:
      patches - 2D numpy array of size patchSize x numPatches
      centroids - 2D numpy array of size patchSize x k
      
    Returns:
      features - 2D numpy array with new feature values for each patch
                 of the image in rows, size is numPatches x k
    """
    k = centroids.shape[1]
    numPatches = patches.shape[1]
    features = np.empty((numPatches,k))

    # BEGIN_YOUR_CODE (around 9 lines of code expected)
    #raise "Not yet implemented"
    # END_YOUR_CODE
    for p in range(numPatches):
        for i in range(k):
            features[p,i] = np.linalg.norm(centroids[:,i] - patches[:,p])
        features[p,:] = np.mean(features[p,:]) - features[p,:]
        for i in range(k):
            if(features[p,i] < 0):
                features[p,i] = 0
    return features

############################################################
# Problem 4.3.1

import math
def logisticGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of the logistic loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of logistic loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    #raise "Not yet implemented."
    # END_YOUR_CODE
    yy = 2*y - 1
    temp = np.exp(-np.dot(theta,featureVector)*yy)
    gradient = [-temp*yy/(1 + temp)]*featureVector
    return gradient
    

############################################################
# Problem 4.3.2
    
def hingeLossGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of hinge loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of hinge loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    #raise "Not yet implemented."
    # END_YOUR_CODE
    yy = 2*y - 1
    if(1 - np.dot(theta,featureVector)*yy > 0):
        gradient = -featureVector*yy
    else:
        gradient = 0*featureVector
    return gradient

