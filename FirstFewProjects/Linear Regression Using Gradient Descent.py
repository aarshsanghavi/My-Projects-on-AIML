import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
x = np.arange(-1,1,0.2)
y = [-0.6,-0.8,-0.4,-0.5,-0.2,-0.35,0,0.2,0.3,0.6]
current_m=1
current_c=1
def predicted_y(x):
    return current_m*x + current_c

sx2,sx1,sy2,sy1,sx1y1=0,0,0,0,0



learning_rate=0.0007
for t in range(0, 10):
    sx2 = sx2 + x[t]**2
    sy2 = sy2 + y[t] ** 2
    sx1=sx1+x[t]
    sy1=sy1+y[t]
    sx1y1=sx1y1 + x[t]*y[t]

def gradient(inpm,inpc):
    return 2*(inpm*sx2 - sx1y1 + inpc*sx1) , 20*inpc - 2*sy1 + 2*inpm*sx1

for _ in range(1000):
    plt.plot(x,predicted_y(x))
    plt.scatter(x,y)
    grm,grc=gradient(current_m,current_c)
    current_m = current_m - grm * learning_rate
    current_c = current_c - grc * learning_rate
    plt.pause(0.001)
    plt.clf()
