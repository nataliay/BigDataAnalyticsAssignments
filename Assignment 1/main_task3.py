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

    
    # We would like to know which hour of the day is the best time for drivers that has the highest 
    # profit per miles.Consider the surcharge amount in dollar for each taxi ride (without tip amount) and 
    # the distance in miles,and sum up the rides for each hour of the day (24 hours) – consider the pickup time 
    # for your calculation.The profit ratio is the ration surcharge in dollar divided by the travel distance in 
    # miles for each specific timeof the day.
    # Profit Ratio = (Surcharge Amount in US Dollar) / (Travel Distance in miles)
    # We are interested to know the time of the day that has the highest profit ratio.

    #Map pickup time (p[2] split twice. first time to extract time and then to extract hour) to their corresponding 
    #miles (p[5]) and surcharge amount (p[12]) then find the summation of surcharge amount and miles per hour
    surchargeAndMiles = taxilinesCorrected.map(lambda p: (p[2].split(' ')[1].split(':')[0], (float(p[5]), float(p[12]))))
    #Filter out rides that have 0 distance and/or have 0 money
    zerosRemoved = surchargeAndMiles.filter(lambda x: x[1][0] != 0 and x[1][1] != 0)
    #Find totals per hour
    sumOfSurchargeAndMiles = zerosRemoved.reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))
    #Find ratio by diving total surcharge amount by total miles for each hour
    profitRatio = sumOfSurchargeAndMiles.map(lambda a: (a[0], round(a[1][1]/a[1][0],2)))
    #Get highest rate and show it as (hour, profitRatio) 
    #Note: profitRatio is rounded to 2 decimal points
    highestRate = sc.parallelize(profitRatio.top(1, lambda x:x[1]))
    #Save to text file
    highestRate.saveAsTextFile(sys.argv[2])
    #highestRate.saveAsTextFile("profitRatio")
    
    
    sc.stop()  

