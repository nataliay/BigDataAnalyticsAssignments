from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import sys

from operator import add
import re
from re import sub, search
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from numpy.random.mtrand import dirichlet, multinomial
from string import punctuation
import random
from pyspark import SparkConf, SparkContext

sc = SparkContext(appName="A6")

#lines = sc.textFile("20-news-same-line.txt")
lines = sc.textFile(sys.argv[1], 1)

regex = re.compile('[^a-zA-Z]')
stopWords = stopwords.words('english')

#Preprocessing:
    #Remove non-letter character
    #Convert all words to lower case
    #Remove words with less than 2 letters
    #Remove stop words
words = lines.map(lambda x : x[x.find(">") : x.find("</doc>")]).flatMap(lambda x: x.split()).\
            map(lambda x: regex.sub('', x).lower()).\
            filter(lambda x: len(x)>2).filter(lambda x: x not in stopWords)

#Count number of times each word is used in all documents
counts = words.map(lambda x: (x, 1)).reduceByKey(add)
counts.cache()

#Find 20,000 most common words
top20kcounts = counts.top(20000, lambda x: x[1])

#Get words without their corresponding count
top20k = []
for pair in top20kcounts:
    top20k.append(pair[0])  

def map_to_array(mapping):
    count_lst = [0] * 20000
    i = 0
    while i < 20000:
        if i in mapping:
            count_lst[i] = mapping[i]
        i+= 1
    return np.array(count_lst)

#This returns (header, dictionary of {ith word rank: ith word count in doc}, number of words in the doc)
def countWords(d):
    try:
        header = search('(<[^>]+>)', d).group(1)
    except AttributeError:
        header = ''
    d = d[d.find(">") : d.find("</doc>")]
    words = d.split(' ')
    numwords = {}
    count = 0
    for w in words:
        if search("([A-Za-z])\w+", w) is not None:
            w = w.lower().strip(punctuation)
            if (len(w) > 2) and w in top20k:
                count += 1
                idx = top20k.index(w)
                if idx in numwords:
                    numwords[idx] += 1
                else:
                    numwords[idx] = 1
    return (header, numwords, count)

result = lines.map(countWords)
result.cache()

print('Number of times that each of the 100 most common dictionary words appear in document 20newsgroups/comp.graphics/37261:\n',
     list(zip(top20k[:100], result.filter(lambda x: 'comp.graphics/37261' in x[0]).map(lambda x: x[1]).map(map_to_array).\
        collect()[0][:100])))
