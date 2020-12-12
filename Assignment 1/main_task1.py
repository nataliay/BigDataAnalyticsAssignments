#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import sys
from operator import add

from pyspark import SparkContext


if __name__ == "__main__":
    
    sc = SparkContext()

    #lines = sc.textFile("taxi-data-sorted-small.csv.bz2")  
    lines = sc.textFile(sys.argv[1])
    taxilines = lines.map(lambda x: x.split(','))

    def isfloat(value):
        try:
            float(value)
            return True
        except:
            return False

    def correctRows(p):
        if (len(p) == 17):
            if (isfloat(p[4]) and isfloat(p[5]) and isfloat(p[11]) and isfloat(p[12]) and isfloat(p[16])):
                if (float(p[4]!=0) and float(p[5]!=0) and float(p[11]!=0) and float(p[12]!=0) and float(p[16])):
                    return p

    taxilinesCorrected = taxilines.filter(correctRows)

    
    # Many different taxis have had multiple drivers.  Write and execute a Spark Python program that 
    # computesthe top ten taxis that have had the largest number of drivers.  
    # Your output should be a set of (medallion,number of drivers) pairs.

    #Get distinct Taxis(p[0])/Drivers(p[1]) combos (count drivers once per taxi)
    distinctTaxiDriver = taxilinesCorrected.map(lambda p: (p[0], p[1]) ).distinct()
    #count number of drivers for each taxi (p[0] from distinctTaxiDriver)
    numbersOfDriversPerTaxis = distinctTaxiDriver.map(lambda p: (p[0], 1)).reduceByKey(lambda a,b: a+b)
    
    #Get top 10 taxis and show them as (medallion,number of drivers) 
    top10Taxis = sc.parallelize(numbersOfDriversPerTaxis.top(10, lambda x:x[1]))
    #Save to text file
    top10Taxis.saveAsTextFile(sys.argv[2])
    #top10Taxis.saveAsTextFile("numbersOfDriversPerTaxis")
    
    
    sc.stop()  

