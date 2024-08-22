import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


learning_rate=0.08
dosage = np.arange(0,1.5,0.1)
rating = [0.0,0.06,0.13,0.28,0.4,0.55,0.62,0.68,0.65,0.60,0.53,0.4,0.32,0.23,0.1]



sx2,sx1,sy2,sy1,sw1,sx1y1=0,0,0,0,0,0
current_m=-1
current_c=1
def predicted_y(x):
    return current_m*x + current_c

print("Enter the dosage where you want the prediction : ")
x= input()
x=float(x)
tau=2
for t in range(0, 15):
    diff = (dosage[t] - x)*10

    weight = np.e**(-1*(diff**2))
    sx2 += (dosage[t]**2)*weight
    sx1 += (dosage[t])*weight
    sx1y1 += (dosage[t]*rating[t])*weight
    sy1 += rating[t]*weight
    sw1 += weight
    """sw1 += weight"""

def gradient(inpm,inpc):
    return 2*(inpm*sx2 - sx1y1 + inpc*sx1) , 2*(inpc*sw1 - sy1 + inpm*sx1)

for t in range(100000):
    grdm , grdc = gradient(current_m,current_c)
    plt.scatter(dosage,rating)
    plt.plot(dosage,predicted_y(dosage))
    plt.pause(0.001)
    plt.clf()
    current_m-=grdm*learning_rate
    current_c-=grdc*learning_rate
