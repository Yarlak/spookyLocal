# -*- coding: utf-8 -*-
"""
Created on Sat Nov 03 16:03:06 2018

@author: OH YEA
"""

import csv
import nltk
import itertools
import time
import numpy as np
import operator
from nltk import FreqDist

def kaggleize(predictions,file):

	if(len(predictions.shape)==1):
		predictions.shape = [predictions.shape[0],1]

	ids = 1 + np.arange(predictions.shape[0])[None].T
	kaggle_predictions = np.hstack((ids,predictions)).astype(int)
	writer = csv.writer(open(file, 'w'))
	writer.writerow(['# id','Prediction'])
	writer.writerows(kaggle_predictions)


def GetFeatures(localWords, allWords):
    features = {}
    for word in allWords:
        features['contains({})'.format(word)] = (word in localWords)
    return features

outputFile = 'NB.csv'

startTime = time.time()
theAuthors = {'EAP':0, 'HPL':1, 'MWS':2}

trainSet = []
allWords = []
testSet = []
idNumber = []
###########################################################################
with open('train.csv', encoding='utf8') as csvfile:
    theReader = csv.reader(csvfile)
    for row in theReader:
        allWords += nltk.word_tokenize(row[1])

tempDist = FreqDist(allWords)

sortedWords = sorted(tempDist.items(), key=operator.itemgetter(1), reverse=True)

allWords = [i[0] for i in sortedWords]

#theWordsSet = set(allWords[500:3000])

tempWords = list(allWords)

theWords = tempWords[:1000]

#theWords = theWordSet[15000:16000]
########################################################################


###########################################################################
with open('train.csv', encoding='utf8') as csvfile:
    theReader = csv.reader(csvfile)
    for row in theReader:
        label = theAuthors[row[2]]
        trainSet.append((GetFeatures(nltk.word_tokenize(row[1]), theWords), label))

with open('test.csv', encoding='utf8') as csvfile:
    theReader = csv.reader(csvfile)
    for row in theReader:
        testSet.append((GetFeatures(nltk.word_tokenize(row[1]), theWords)))
        idNumber.append(row[0])
    
splitNum = int(round(len(trainSet)*0.8))

theClass = nltk.NaiveBayesClassifier.train(trainSet)

theAccuracy = nltk.classify.accuracy(theClass, trainSet[splitNum:])
theClass.show_most_informative_features(10)

printCount = 0

thePredictions = []

for item in testSet:
	thePredictions.append(nltk.NaiveBayesClassifier.prob_classify(theClass, item))

with open('accuracy.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')    
    writer.writerow([theAccuracy])


with open('names.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')    
    writer.writerow(['id', 'EAP', 'HPL', 'MWS'])
    

    for theThing in thePredictions:
        prediction = theThing
        tempList = []
        tempList.append(idNumber[printCount])
        for label in prediction.samples():
            tempList.append(prediction.prob(label))
        printCount += 1
        writer.writerow(tempList)

