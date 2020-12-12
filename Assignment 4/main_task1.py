import sys
import re
import numpy as np

from numpy import dot
from numpy.linalg import norm

from operator import add
from pyspark import SparkContext

def freqArray (listOfIndices, numberofwords):
    returnVal = np.zeros (20000)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    returnVal = np.divide(returnVal, numberofwords)
    return returnVal

def build_zero_one_array (listOfIndices):
    returnVal = np.zeros (20000)
    for index in listOfIndices:
        if returnVal[index] == 0: returnVal[index] = 1
    return returnVal

sc = SparkContext(appName="LogisticRegression")

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
allDocsAsNumpyArraysTrain = allDictionaryWordsInEachDocTrain.map(lambda x: (x[0], 
                                                                  freqArray (x[1], len(x[1]))))
allDocsAsNumpyArraysTrain.cache()
# Now, create a version of allDocsAsNumpyArrays where, in the array,
# every entry is either zero or one.
# A zero means that the word does not occur,
# and a one means that it does.
zeroOrOneTrain = allDocsAsNumpyArraysTrain.map(lambda x: (x[0], build_zero_one_array(np.where(x[1]>0)[0])))

# Now, add up all of those arrays into a single array, where the
# i^th entry tells us how many
# individual documents the i^th word in the dictionary appeared in
dfArrayTrain = zeroOrOneTrain.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]

# Get the version of dfArray where the i^th entry is the inverse-document frequency for the
# i^th word in the corpus
idfArrayTrain = np.log(np.divide(np.full(20000, numberOfDocsTrain), dfArrayTrain))

# Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
allDocsAsNumpyArraysTFidfTrain = allDocsAsNumpyArraysTrain.map(lambda x: 
                                    (1 if 'AU' in x[0] else 0, np.multiply(x[1], idfArrayTrain)))
allDocsAsNumpyArraysTFidfTrain.cache()

wikiTF = allDocsAsNumpyArraysTrain.filter(lambda x: 'AU' not in x[0])
AusTF = allDocsAsNumpyArraysTrain.filter(lambda x: 'AU' in x[0])

#applicant
pos = dictionary.filter(lambda x: x[0] == 'applicant').take(1)[0][1]
print('Average TF for "applicant" in Wikipedia:', wikiTF.map(lambda x: (x[1][pos])).mean())
#and
pos = dictionary.filter(lambda x: x[0] == 'and').take(1)[0][1]
print('Average TF for "and" in Wikipedia:', wikiTF.map(lambda x: (x[1][pos])).mean())
#attack
pos = dictionary.filter(lambda x: x[0] == 'attack').take(1)[0][1]
print('Average TF for "attack" in Wikipedia', wikiTF.map(lambda x: (x[1][pos])).mean())
#protein
pos = dictionary.filter(lambda x: x[0] == 'protein').take(1)[0][1]
print('Average TF for "protein" in Wikipedia', wikiTF.map(lambda x: (x[1][pos])).mean())
#court
pos = dictionary.filter(lambda x: x[0] == 'court').take(1)[0][1]
print('Average TF for "court" in Wikipedia', wikiTF.map(lambda x: (x[1][pos])).mean(), '\n')

#applicant
pos = dictionary.filter(lambda x: x[0] == 'applicant').take(1)[0][1]
print('Average TF for "applicant" in Australian Court:', AusTF.map(lambda x: (x[1][pos])).mean())
#and
pos = dictionary.filter(lambda x: x[0] == 'and').take(1)[0][1]
print('Average TF for "and" in Australian Court:', AusTF.map(lambda x: (x[1][pos])).mean())
#attack
pos = dictionary.filter(lambda x: x[0] == 'attack').take(1)[0][1]
print('Average TF for "attack" in Australian Court', AusTF.map(lambda x: (x[1][pos])).mean())
#protein
pos = dictionary.filter(lambda x: x[0] == 'protein').take(1)[0][1]
print('Average TF for "protein" in Australian Court', AusTF.map(lambda x: (x[1][pos])).mean())
#court
pos = dictionary.filter(lambda x: x[0] == 'court').take(1)[0][1]
print('Average TF for "court" in Australian Court', AusTF.map(lambda x: (x[1][pos])).mean())

sc.stop()