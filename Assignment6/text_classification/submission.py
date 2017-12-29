"""
Text classification
"""

import util
import operator
from collections import Counter

class Classifier(object):
    def __init__(self, labels):
        """
        @param (string, string): Pair of positive, negative labels
        @return string y: either the positive or negative label
        """
        self.labels = labels

    def classify(self, text):
        """
        @param string text: e.g. email
        @return double y: classification score; >= 0 if positive label
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, text):
        """
        @param string text: the text message
        @return string y: either 'ham' or 'spam'
        """
        if self.classify(text) >= 0.:
            return self.labels[0]
        else:
            return self.labels[1]

class RuleBasedClassifier(Classifier):
    def __init__(self, labels, blacklist, n=1, k=-1):
        """
        @param (string, string): Pair of positive, negative labels
        @param list string: Blacklisted words
        @param int n: threshold of blacklisted words before email marked spam
        @param int k: number of words in the blacklist to consider
        """
        super(RuleBasedClassifier, self).__init__(labels)
        # BEGIN_YOUR_CODE (around 3 lines of code expected)
        if k > 0: self.blacklist = set(blacklist[0:k])
        else: self.blacklist = set(blacklist)
        self.n = n
        self.k = k
        #raise NotImplementedError("TODO:")
        # END_YOUR_CODE

    def classify(self, text):
        """
        @param string text: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        words = text.split()
        count = 0
        for word in words:
            if word in self.blacklist: count += 1
            if count > self.n: return -1.0
        return 1.0
        #raise NotImplementedError("TODO:")
        # END_YOUR_CODE

def extractUnigramFeatures(x):
    """
    Extract unigram features for a text document $x$.
    @param string x: represents the contents of an text message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    words = x.split()
    count = Counter()
    for word in words:
        count[word] += 1
    return dict(count)
    #raise NotImplementedError("TODO:")
    # END_YOUR_CODE


class WeightedClassifier(Classifier):
    def __init__(self, labels, featureFunction, params):
        """
        @param (string, string): Pair of positive, negative labels
        @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
        @param dict params: the parameter weights used to predict
        """
        super(WeightedClassifier, self).__init__(labels)
        self.featureFunction = featureFunction
        self.params = params

    def classify(self, x):
        """
        @param string x: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        y, features = 0.0, self.featureFunction(x)
        for ele in self.params:
            if ele in features: y += features[ele] * self.params[ele]
        return y
        #raise NotImplementedError("TODO:")
        # END_YOUR_CODE

def learnWeightsFromPerceptron(trainExamples, featureExtractor, labels, iters = 20):
    """
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @params func featureExtractor: Function to extract features, e.g. extractUnigramFeatures
    @params labels: tuple of labels ('pos', 'neg'), e.g. ('spam', 'ham').
    @params iters: Number of training iterations to run.
    @return dict: parameters represented by a mapping from feature (string) to value.
    """
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    #weights = 0.0
    print "this takes some time..."
    params = {}
    for niter in range(iters):
        for x,y in trainExamples:
            y = 1 if y == labels[0] else -1
            features = featureExtractor(x)
            y_hat = 0.0
            for ele in features:
                if ele not in params: params[ele] = 0.0
                else: y_hat += features[ele] * params[ele]
            y_hat = 1 if y_hat >= 0.0 else -1
            if y_hat != y:
                for ele in features:
                    params[ele] += y * features[ele]
    return params
    #raise NotImplementedError("TODO:")
    # END_YOUR_CODE

def extractBigramFeatures(x):
    """
    Extract unigram + bigram features for a text document $x$.

    @param string x: represents the contents of an email message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 12 lines of code expected)
    token = '-BEGIN-'
    words = x.split()
    count = Counter()
    for i, word in enumerate(words):
        if i == 0: biword = token + ' ' + word
        else: biword = words[i-1] + ' ' + word
        count[biword] += 1
        count[word] += 1
    return dict(count)
    #raise NotImplementedError("TODO:")
    # END_YOUR_CODE

class MultiClassClassifier(object):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); each classifier is a WeightedClassifier that detects label vs NOT-label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        self.labels = labels
        self.classifiers = classifiers
        #raise NotImplementedError("TODO:")
        # END_YOUR_CODE

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, x):
        """
        @param string x: the text message
        @return string y: one of the output labels
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        scores = self.classify(x)
        return max(scores, key = operator.itemgetter(1))[0]
        #raise NotImplementedError("TODO:")
        # END_YOUR_CODE

class OneVsAllClassifier(MultiClassClassifier):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); the classifier is the one-vs-all classifier
        """
        super(OneVsAllClassifier, self).__init__(labels, classifiers)

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        scores = []
        for label, classifier in self.classifiers:
            scores.append((label, classifier.classify(x)))
        return scores
        #raise NotImplementedError("TODO:")
        # END_YOUR_CODE

def learnOneVsAllClassifiers( trainExamples, featureFunction, labels, perClassifierIters = 10 ):
    """
    Split the set of examples into one label vs all and train classifiers
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label (an entry from the list of labels)
    @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
    @param list string labels: List of labels
    @param int perClassifierIters: number of iterations to train each classifier
    @return list (label, Classifier)
    """
    # BEGIN_YOUR_CODE (around 10 lines of code expected)
    classifiers = []
    for label in labels:
        #classifier = WeightedClassifier(labels, featureFunction, params)
        newTrainEx = []
        for x,y in trainExamples:
            y = label if y == label else "NOT"
            newTrainEx.append((x, y))
        new_labels = (label, "NOT")
        params = learnWeightsFromPerceptron(newTrainEx, featureFunction, new_labels, perClassifierIters)
        classifier = WeightedClassifier(new_labels, featureFunction, params)
        classifiers.append((label, classifier))
    return classifiers
    #raise NotImplementedError("TODO:")
    # END_YOUR_CODE
