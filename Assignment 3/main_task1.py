from __future__ import print_function

import sys
from operator import add
import numpy as np

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
            if (isfloat(p[5]) and isfloat(p[11]) and isfloat(p[4]) and isfloat(p[15])):
                if (float(p[4])>= 2*60 and float(p[4])<= 60*60
                    and float(p[11]) >= 3 and float(p[11]) <= 200 
                    and float(p[5]) >= 1 and float(p[5]) <= 50
                    and float(p[15]) >= 3):
                    return p

    
taxilinesCorrected = taxilines.filter(correctRows)

taxilinesCorrected = taxilinesCorrected.map(lambda x: (float(x[5]), float(x[11]), 
                                                           float(x[5])*float(x[11]), 
                                                           float(x[5])**2))
n = float(taxilinesCorrected.count())
print("Number of observations: ", n)

sum_x = taxilinesCorrected.map(lambda x: x[0]).reduce(add)
sum_y = taxilinesCorrected.map(lambda x: x[1]).reduce(add)
sum_xy = taxilinesCorrected.map(lambda x: x[2]).reduce(add)
sum_x2 = taxilinesCorrected.map(lambda x: x[3]).reduce(add)

m = ((n * sum_xy) - (sum_x * sum_y)) / ((n * sum_x2) - (sum_x**2))
b = ((sum_x2 * sum_y) - (sum_x * sum_xy)) / ((n * sum_x2) - (sum_x**2))
print('m = ', round(m, 2), ' b = ', round(b, 2))

sc.stop()