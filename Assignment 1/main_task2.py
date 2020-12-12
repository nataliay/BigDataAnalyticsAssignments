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

    
        # We would like to figure out who the top 10 best drivers are in terms of their average earned money 
    # perminute spent carrying a customer. The total amount field is the total money earned on a trip. 
    # In the end, we are interested in computing a set of (driver, money per minute) pairs

    #Map drivers IDs (p[1]) to their corresponding minutes (p[4]/60) and total amount (p[16]) then find the 
    #summation of minutes and total amounts per driver ID
    minsAndMoney = taxilinesCorrected.map(lambda p: (p[1], (float(p[4])/60, float(p[16]))))
    #Filter out rides that are less than 1 minute and/or have 0 money
    zerosRemoved = minsAndMoney.filter(lambda x:x[1][0] >= 1 and x[1][1] != 0)
    #Find totals per driver
    sumOfMinsAndMoney = zerosRemoved.reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))
    #Find ratio by diving total amount by total minutes for each driver
    moneyPerMinute = sumOfMinsAndMoney.map(lambda a: (a[0], round(a[1][1]/a[1][0],2)))
    #Get top 10 drivers and show them as (driver, money per minute) 
    #Note: Money per minute is rounded to 2 decimal points
    top10drivers = sc.parallelize(moneyPerMinute.top(10, lambda x:x[1]))
    #Save to text file
    top10drivers.saveAsTextFile(sys.argv[2])
    #top10drivers.saveAsTextFile("moneyPerMinute")
    
    
    sc.stop()  

