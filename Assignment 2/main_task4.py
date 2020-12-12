#!/usr/bin/env python
# coding: utf-8

import sys
import re
import numpy as np

from numpy import dot
from numpy.linalg import norm

from operator import add
from pyspark import SparkContext, SQLContext
from pyspark.sql import functions as func
from pyspark.sql.functions import udf, monotonically_increasing_id
from pyspark.sql.types import *


sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)



# Set the file paths on your local machine
# Change this line later on your python script when you want to run this on the CLOUD (GC or AWS)

#wikiPagesFile= "WikipediaPagesOneDocPerLine1000LinesSmall.txt.bz2"
wikiPagesFile = sys.argv[1]
#wikiCategoryFile="wiki-categorylinks-small.csv.bz2"
wikiCategoryFile = sys.argv[2]



# Read two files into DFs

wikiCats=sqlContext.read.format('csv')\
.options(header = 'false', inferSchema = 'true', sep = ',')\
.load(wikiCategoryFile)

# Now the wikipages
wikiPages = sqlContext.read.format('csv')\
.options(header = 'false', inferSchema = 'true', sep = '|')\
.load(wikiPagesFile)


# Assumption: Each document is stored in one line of the text file
# We need this count later ... 
numberOfDocs = wikiPages.count()

# Each entry in validLines will be a line from the text file
validLines = wikiPages.filter(wikiPages['_c0'].contains('id' and 'url='))

# Now, we transform it into a set of (docID, text) pairs

#UDF to index the key and another one to index the text
find_index_key = udf(lambda x : x[x.find('id="') + 4 : x.find('" url=')])
find_index_text = udf(lambda x : x[x.find('">') + 2:-6])
#Apply above udf to create two new columns (key and text)
keyAndText = validLines.withColumn('key', find_index_key('_c0'))
keyAndText = keyAndText.withColumn('text', find_index_text('_c0')).select('key', 'text')


def buildArray(listOfIndices):
    
    returnVal = np.zeros(20000)
    
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    
    mysum = np.sum(returnVal)
    
    returnVal = np.divide(returnVal, mysum)
    
    return returnVal.tolist()

buildArray_udf = udf(buildArray, ArrayType(FloatType(), containsNull=False))

def cosinSim (x,y):
	normA = np.linalg.norm(x)
	normB = np.linalg.norm(y)
	return (np.dot(x,y)/(normA*normB)).tolist()

cosinSim_udf = udf(cosinSim, FloatType())

multiply_idfArray = udf(lambda x: np.multiply(x, idfArray).tolist(), ArrayType(FloatType(), containsNull=False))


# Now, we split the text in each (docID, text) pair into a list of words
# After this step, we have a data set with
# (docID, ["word1", "word2", "word3", ...])
# We use a regular expression here to make
# sure that the program does not break down on some of the documents

# remove all non letter characters
keyAndListOfWords = keyAndText.withColumn ('wordsSet', func.split(func.lower(
    func.regexp_replace('text', '[^a-zA-Z]', ' ')), ' ')).select('key', 'wordsSet')
# better solution here is to use NLTK tokenizer

# Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
#To transform it to ("word1", 1) ("word2", 1)..., Use explode to put each word for each 
#document in different rows then group by words and then use agg to count word
allWords = keyAndListOfWords.withColumn('words', func.explode('wordsSet')).drop('key', 'wordsSet')


# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.groupBy('words').agg(func.count('words')).filter('words != ""')

# Get the top 20,000 words in a local array in a sorted format based on frequency
# If you want to run it on your laptio, it may a longer time for top 20k words. 
topWords = allCounts.orderBy('count(words)', ascending = False).limit(20000)

print("Top Words in Corpus:", allCounts.orderBy('count(words)', ascending = False).limit(10).collect())

# We'll create a DF that has a set of (word, dictNum) pairs
# Add a new column that contains increasing_id (it will create ids from 0 to the length of 
# topWords (20000)) so the dictionary DF will have a words column from topWords and the newly created 
# dictNum column
# the number will be the spot in the dictionary used to tell us
# where the word is located
dictionary = topWords.withColumn('dictNum', monotonically_increasing_id()).drop('count(words)')

print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", 
      dictionary.orderBy('dictNum', ascending = False).limit(20).collect())


################### TASK 2  ##################

# Next, we get a DF that has, for each (docID, ["word1", "word2", "word3", ...]),
# ("word1", docID), ("word2", docId), ...
allWordsWithDocID = keyAndListOfWords.withColumn('words', func.explode('wordsSet')).drop('wordsSet').\
filter('words != ""')

# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWords = dictionary.join(allWordsWithDocID, 'words')

# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPos = allDictionaryWords.drop('words')

# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
# Group by key and use collect_set to combine dictNum for each key
allDictionaryWordsInEachDoc = justDocAndPos.groupBy('key').agg(func.collect_set('dictNum'))

# The following line this gets us a set of
# (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
# and converts the dictionary positions to a bag-of-words numpy array...
# use the buildArray function to build the feature array
# regexp_replace is used to remove square brackets from buildArray output
allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.withColumn('NumpyArray',buildArray_udf(
    'collect_set(dictNum)')).drop('collect_set(dictNum)')

print(allDocsAsNumpyArrays.show(3))

#To find dfArray, use justDocAndPos
#First remove duplicate rows (a word might be in the same doc multiple times)
#Then group by dictNum then count docs (key) for each dictNum
#Convert to numpy array, flatten and sort it when done

# the i^th entry tells us how many
# individual documents the i^th word in the dictionary appeared in
dfArray = np.array(justDocAndPos.distinct().groupBy('dictNum').agg(func.count('key')).\
                   drop('dictNum').collect()).flatten()

# Get the version of dfArray where the i^th entry is the inverse-document frequency for the
# i^th word in the corpus
idfArray = np.log(np.divide(np.full(20000, numberOfDocs), dfArray))

# Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors and remove []
# Then split by ,
allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.withColumn('TFidf', multiply_idfArray(
    'NumpyArray')).drop('NumpyArray')

print(allDocsAsNumpyArraysTFidf.show(2))

# Now, we join it with categories, and map it after join so that we have only the wikipageID 
# This joun can take time on your laptop. 
# You can do the join once and generate a new wikiCats data and store it. Our WikiCategories includes all categories
# of wikipedia. 

wikiAndCatsJoind = wikiCats.withColumnRenamed('_c0', 'key').withColumnRenamed('_c1', 'category').\
            join(allDocsAsNumpyArraysTFidf, on='key')
featuresDF = wikiAndCatsJoind.select('category', 'TFidf')

# Cache this important data because we need to run kNN on this data set. 
featuresDF.cache()
featuresDF.show(10)

# Finally, we have a function that returns the prediction for the label of a string, using a kNN algorithm
def getPrediction (textInput, k):
    # Create an DF out of the textIput
    myDoc = sqlContext.createDataFrame([textInput], StringType())
    
    #Flat map the text to (word, 1) pair for each word in the doc
    wordsInThatDoc = myDoc.withColumn ('words', func.explode(func.split(func.lower(
                                                func.regexp_replace('value', '[^a-zA-Z]', ' ')), ' '))).\
                                                withColumn('count', func.lit(1)).\
                                                filter('words != ""')

    # This will give us a set of (word, (dictionaryPos, 1)) pairs
    allDictionaryWordsInThatDoc = dictionary.join (wordsInThatDoc, on='words').\
                                select('dictNum', 'count').groupBy('count').\
                                    agg(func.collect_set('dictNum'))
    
    #Get tf array for the input string
    myArray = allDictionaryWordsInThatDoc.orderBy('count', ascending = False).limit(1)\
                        .withColumn('myArray', buildArray_udf('collect_set(dictNum)')).\
                        select('myArray')
    # Multiply by idfArray
    myArray = myArray.withColumn('myArrayxIdfArray', multiply_idfArray('myArray')).select('myArrayxIdfArray')
    
    # Get the tf * idf array for the input string
    # Get the distance from the input text string to all database documents, 
    # using cosine similarity (np.dot() )
    distances = featuresDF.join(myArray)
    distances = distances.withColumn('distances', cosinSim_udf('TFidf', 'myArrayxIdfArray')).\
                            select('category', 'distances')

    
    # get the top k distances
    topK = distances.orderBy('distances', ascending = False).limit(k)

    # now, for each docID, get the count of the number of times this document ID appeared in the top k
    numTimes = topK.groupBy('category').agg(func.count('category').alias('count')).\
                                                                            drop('distances')

    # Return the top 1 of them.
    # Ask yourself: Why we are using twice top() operation here?
    # Answer: to show them in sorted order
    return numTimes.orderBy('count', ascending = False).limit(k).collect()
# In[21]:


print(getPrediction('Sport Basketball Volleyball Soccer', 10))


# In[22]:


print(getPrediction('What is the capital city of Australia?', 10))


# In[23]:


print(getPrediction('How many goals Vancouver score last year?', 10))


# In[ ]:


# Congradulations, you have implemented a prediction system based on Wikipedia data. 
# You can use this system to generate automated Tags or Categories for any kind of text 
# that you put in your query.
# This data model can predict categories for any input text. 

