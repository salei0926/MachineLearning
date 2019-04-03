import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy import *
np.seterr(divide='ignore',invalid='ignore')

p_x = np.array([[3,3],[2,3],[1,1],[3,2]])
y = np.array([1,1,-1,-1])
plt.figure()
for i in range(len(p_x)):
    if y[i]==1:
        plt.plot(p_x[i][0],p_x[i][1],'ro')
    else:
        plt.plot(p_x[i][0],p_x[i][1],'bo')

w = np.array([0,0])
b = 0
delta = 1
for i in range (500):

    for j in range(len(p_x)):
        if (y[j]*(np.dot(w,p_x[j]) + b))<=0:
            w = w + delta * y[j] * p_x[j]
            b = b + delta * y[j]
print(w,b)
line_x = np.arange(0,6)
line_y= (-w[0]*line_x - b)/w[1]
print(line_x,line_y)
plt.plot(line_x,line_y)
plt.show()


















