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

taxilinesCorrected = taxilinesCorrected.map(lambda x: 
                                (np.array([1, float(x[4])/60, float(x[5]), float(x[11]), float(x[12])]),
                                float(x[16])))

n = float(taxilinesCorrected.count())
print("Number of observations: ", n)
taxilinesCorrected.cache()

theta = np.repeat(0.1, 5)
lr = 0.001
num_iteration = 100 
precision = 0.1
cost_old = 0

#Iterate a maximum of 100 times
for i in range(num_iteration):
    
    #Calculate predicted y and map it to the RDD
    xy_ypred = taxilinesCorrected.map(lambda x: ((x[0]), float(x[1]), (sum(theta * (x[0])))))
    
    #Calculate cost (sum(y - (mx+b))^2)/n
    cost = (xy_ypred.map(lambda x: ((x[1] - x[2])**2)).reduce(add))/n

    #Calculate derivatives 
    theta_deriv = (-2.0/n) * xy_ypred.map(lambda x: (x[0] * (x[1] - x[2]))).reduce(add)

    #Bold driver (starts from second iteration and assumes lr = 0.001 for first one)
    #If cost decreases, increment lr
    if (i >= 1):
        if(cost < cost_old):
            lr = lr * 1.05
        #Else if cost increases, decrease lr
        else:
            lr = lr * 0.5
        
    #Update theta using lr and new derivatives 
    theta = theta - lr * theta_deriv
    
    #If difference between new cost and previous cost is <= 0.1, stop the loop (cost is not decreasing) 
    if(abs(cost - cost_old) <= precision):
        print('Cost stopped decreasing at: ', i)
        break
    
    cost_old = cost
    
    print('Iteration = ', i , ' Cost = ', cost, ' lr = ', lr,
          ' theta = ', np.round(theta, 2))

sc.stop()
