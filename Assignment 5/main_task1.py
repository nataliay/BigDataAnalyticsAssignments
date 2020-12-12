import sys
import re
import numpy as np

from numpy import dot
from numpy.linalg import norm

from operator import add
from pyspark import SparkContext

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint

from datetime import datetime

def freqArray (listOfIndices, numberofwords):
    returnVal = np.zeros (20000)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    returnVal = np.divide(returnVal, numberofwords)
    return returnVal

sc = SparkContext()

now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print("Start Time: ", start_time)

now = datetime.now()
train_read_start_time = now.strftime("%H:%M:%S")
print("Reading training data started at: ", train_read_start_time)

#d_corpusTrain = sc.textFile('SmallTrainingData.txt') 
d_corpusTrain = sc.textFile(sys.argv[1], 1)

numberOfDocsTrain = d_corpusTrain.count()

d_keyAndTextTrain = d_corpusTrain.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], 
                                        x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')

d_keyAndListOfWordsTrain = d_keyAndTextTrain.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = d_keyAndListOfWordsTrain.flatMap(lambda x: x[1]).map(lambda x: (x, 1))

# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey(add)

# Get the top 20,000 words in a local array in a sorted format based on frequency
# If you want to run it on your laptio, it may a longer time for top 20k words. 
topWords = allCounts.top(20000, lambda x: x[1])

# We'll create a RDD that has a set of (word, dictNum) pairs
# start by creating an RDD that has the number 0 through 20000
# 20000 is the number of words that will be in our dictionary
topWordsK = sc.parallelize(range(20000))

# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
# ("NextMostCommon", 2), ...
# the number will be the spot in the dictionary used to tell us
# where the word is located
dictionary = topWordsK.map (lambda x : (topWords[x][0], x))
dictionary.cache()

# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
# ("word1", docID), ("word2", docId), ...
allWordsWithDocIDTrain = d_keyAndListOfWordsTrain.flatMap(lambda x: ((j, x[0]) for j in x[1]))

# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWordsTrain = dictionary.join(allWordsWithDocIDTrain)

# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPosTrain = allDictionaryWordsTrain.map(lambda x: (x[1][1], x[1][0]))

# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDocTrain = justDocAndPosTrain.groupByKey()

# The following line this gets us a set of
# (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
# and converts the dictionary positions to a bag-of-words numpy array...
allDocsAsNumpyArraysTrain = allDictionaryWordsInEachDocTrain.map(lambda x: (1 if 'AU' in x[0] else 0, 
                                                                  freqArray (x[1], len(x[1]))))
allDocsAsNumpyArraysTrain.cache()

now = datetime.now()
train_read_end_time = now.strftime("%H:%M:%S")
print("Reading training data ended at: ", train_read_end_time)

now = datetime.now()
train_start_time = now.strftime("%H:%M:%S")
print("Training started at: ", train_start_time)

#Load and parse the data
def parsePoint(line):
    return LabeledPoint(line[0], line[1])
allDocsAsNumpyArraysTrainParsed = allDocsAsNumpyArraysTrain.map(parsePoint)

# Build the model
model = LogisticRegressionWithLBFGS.train(allDocsAsNumpyArraysTrainParsed, iterations = 100)

now = datetime.now()
train_end_time = now.strftime("%H:%M:%S")
print("Training ended at: ", train_end_time)

now = datetime.now()
test_read_start_time = now.strftime("%H:%M:%S")
print("Reading testing data started at: ", test_read_start_time)

#d_corpusTest = sc.textFile('Testing.txt') 
d_corpusTest = sc.textFile(sys.argv[2], 1)

numberOfDocsTest = d_corpusTest.count()

d_keyAndTextTest = d_corpusTest.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], 
                                        x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')

d_keyAndListOfWordsTest = d_keyAndTextTest.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
# ("word1", docID), ("word2", docId), ...
allWordsWithDocIDTest = d_keyAndListOfWordsTest.flatMap(lambda x: ((j, x[0]) for j in x[1]))

# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWordsTest = dictionary.join(allWordsWithDocIDTest)

# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPosTest = allDictionaryWordsTest.map(lambda x: (x[1][1], x[1][0]))

# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDocTest = justDocAndPosTest.groupByKey()

# The following line this gets us a set of
# (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
# and converts the dictionary positions to a bag-of-words numpy array...
allDocsAsNumpyArraysTest = allDictionaryWordsInEachDocTest.map(lambda x: (1 if 'AU' in x[0] else 0, 
                                                                  freqArray (x[1], len(x[1]))))
allDocsAsNumpyArraysTest.cache()

now = datetime.now()
test_read_end_time = now.strftime("%H:%M:%S")
print("Reading testing data ended at: ", test_read_end_time)

now = datetime.now()
test_start_time = now.strftime("%H:%M:%S")
print("Testing started at: ", test_start_time)

allDocsAsNumpyArraysTestParsed = allDocsAsNumpyArraysTest.map(parsePoint)

ytest_ypred = allDocsAsNumpyArraysTestParsed.map(lambda x: (x.label, model.predict(x.features)))

#Positive: 1 - Aus
#Negative: 0 - Wiki    
tp = ytest_ypred.filter(lambda x: x[0] == 1 and x[1] == 1).count()
fp = ytest_ypred.filter(lambda x: x[0] == 0 and x[1] == 1).count()
fn = ytest_ypred.filter(lambda x: x[0] == 1 and x[1] == 0).count()
f1 = tp / (tp + 0.5*(fp+fn))
print('F1 score: ', round(f1*100, 2), '%')

now = datetime.now()
test_end_time = now.strftime("%H:%M:%S")
print("Testing ended at: ", test_end_time)

now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print("End Time: ", end_time)

sc.stop()