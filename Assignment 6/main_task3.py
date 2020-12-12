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

alpha = [0.1] * 20
beta = np.array([0.1] * 20000)

pi = dirichlet(alpha).tolist() # *** vector gives prevalence of each category
mu = np.array([dirichlet(beta) for j in range(20)]) # *** prob vector prevelence of each word of category in each doc 
log_mu = np.log(mu)
header = result.map(lambda x: x[0]).collect()
x = result.map(lambda x: x[1]).map(map_to_array).cache() # *** Num of occurance of each word in each doc. 

# getProbs accepts four parameters:
#
# checkParams: set to true if you want a check on all of the params
#   that makes sure that everything looks OK. This will make the
#   function run slower; use only for debugging
#
# x: 1-D NumPy array, where j^th entry is the number of times that
#   word j was observed in this particular document
#
# pi: the vector of probabilities that tells us how prevalent each doc
#   class is in the corpus
#
# log_allMus: NumPy matrix, where each row is associated with
#   a different document class. A row gives us the list of log-word
#   probabilities associated with the document class.
#
# returns: a NumPy vector, where the j^th entry in the vector is the
#   probability that the document came from each of the different
#   classes
#
def getProbs (checkParams, x, pi, log_allMus):
    #
    if checkParams == True:
            if x.shape [0] != log_allMus.shape [1]:
                    raise Exception ('Number of words in doc does not match')
            if pi.shape [0] != log_allMus.shape [0]:
                    raise Exception ('Number of document classes does not match')
            if not (0.999 <= np.sum (pi) <= 1.001):
                    raise Exception ('Pi is not a proper probability vector')
            for i in range(log_allMus.shape [0]):
                    if not (0.999 <= np.sum (np.exp (log_allMus[i])) <= 1.001):
                            raise Exception ('log_allMus[' + str(i) + '] is not a proper probability vector')
    #
    # to ensure that we don’t have any underflows, we will do
    # all of the arithmetic in “log space”. Specifically, according to
    # the Multinomial distribution, in order to compute
    # Pr[x | class j] we need to compute:
    #
    #       pi[j] * prod_w allMus[j][w]^x[w]
    #
    # If the doc has a lot of words, we can easily underflow. So
    # instead, we compute this as:
    #
    #       log_pi[j] + sum_w x[w] * log_allMus[j][w]
    #
    allProbs = np.log (pi)
    #
    # consider each of the classes, in turn
    for i in range(log_allMus.shape [0]):
            product = np.multiply (x, log_allMus[i])
            allProbs[i] += np.sum (product)
    #
    # so that we don’t have an underflow, we find the largest
    # logProb, and then subtract it from everyone (subtracting
    # from everything in an array of logarithms is like dividing
    # by a constant in an array of “regular” numbers); since we
    # are going to normalize anyway, we can do this with impunity
    #
    biggestLogProb = np.amax (allProbs)
    allProbs -= biggestLogProb
    #
    # finally, get out of log space, and return the answer
    #
    allProbs = np.exp (allProbs)
    return allProbs / np.sum (allProbs)

# *** Gibbs sampling 200x
# *** Gibbs sampling 200x
for num_iter in range(200):
    print(num_iter)
    # update c
    logPi = np.log(pi)

    probs = x.map(lambda x_i : getProbs(False, log_mu, x_i, logPi))
    # Now we need to asign and find out to which category goes each document

    c = probs.map(lambda prob: np.nonzero(multinomial(1, prob))[0][0]) # *** c is the assignment of each doc to a category

    # update pi 
    count = dict(c.map(lambda cat : (cat, 1)).reduceByKey(add).takeOrdered(20)) # *** this is 'a' (vector of size 20) in the PDF

    # Now, we update the alpha 
    new_alpha = [0] * 20
    for i in range(20):
        if i in count:
            new_alpha[i] = alpha[i] + count[i] # *** count[i],  where i is the key
        else:
            new_alpha[i] = alpha[i]

	# use the new alpha to take samples from dirichlet. # *** posterior conjugate (PDF)
    pi = dirichlet(new_alpha)
    
    # update mu
    x_c = x.zip(c).cache() # *** Maybe c is the category so zips docs to catg number

    # generate an empty RDD with all zeros 
    empty = sc.parallelize([np.array([0]*20000)])

    # *** Updates the mean 
    for j in range(20):
        count = x_c.filter(lambda term : term[1] == j) \
                .map(lambda term : term[0]) \
                .union(empty).reduce(add)

        log_mu[j] = np.log(dirichlet(np.add(beta, count))) # **** Spread of each word in each cat in each doc. Count is the (b). 

tosave = []
for mu in log_mu:
    mylist = zip(top20k, mu.tolist())
    mylist = list(mylist)
    mylist = sorted(mylist, key=(lambda x : x[1]))
    tosave.append(list(map((lambda y: y[0]), mylist))[:50])

print('50 most important words in each of the 20 mixture components:\n', tosave)
