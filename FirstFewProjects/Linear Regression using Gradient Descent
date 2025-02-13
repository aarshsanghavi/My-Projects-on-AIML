import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





m1=m3=c3=1
c1=1
c2=1
m2=1
m4=1
learning_rate=0.004
dosage = np.arange(0,1.5,0.1)
rating = [0.0,0.06,0.13,0.28,0.4,0.55,0.62,0.68,0.65,0.60,0.53,0.4,0.32,0.23,0.1]


def soft1_fun(x):
    return np.log(1+np.e**(m1*x+c1))

def soft2_fun(x):
    return np.log(1+np.e**(m2*x+c2))

def bigm1_fun(x):
    return (m3*x*(np.e**(m1*x+c1)))/(1+np.e**(m1*x+c1))

def bigm2_fun(x):
    return (m4*x*(np.e**(m2*x+c2)))/(1+np.e**(m2*x+c2))

def bigc1_fun(x):
    return (m3*(np.e**(m1*x+c1)))/(1+np.e**(m1*x+c1))

def bigc2_fun(x):
    return (m4*(np.e**(m2*x+c2)))/(1+np.e**(m2*x+c2))

def sum_fun(x):
    return (soft1_fun(x)*m3)+(soft2_fun(x)*m4)+c3

logm1c1 = logm2c2 = logm1c12 = logm2c22 = logm1c1m2c2 = bigm1 = bigm2 = bigc1 = bigc2 = logm1c1y = logm2c2y = bigm1y = bigm2y = bigc1y = bigc2y = Y = bigm1logm1c1 = bigm2logm2c2 = bigm1logm2c2 = bigm2logm1c1 = bigc1logm1c1 = bigc1logm2c2 = bigc2logm1c1 = bigc2logm2c2 = 0

def gradient_c3():
    return 2*(m3*logm1c1 + m4*logm2c2 + 14*c3 - Y)

def gradient_m3():
    return 2*(m3*logm1c12 + m4*logm1c1m2c2 + c3*logm1c1 + logm1c1y)

def gradient_m4():
    return 2 * (m3 * logm1c1m2c2 + m4 * logm2c22 + c3 * logm2c2 + logm2c2y)

def gradient_m1():
    return 2 * (m3*bigm1logm1c1 + m4*bigm1logm2c2 + c3*bigm1 - bigm1y)

def gradient_m2():
    return 2 * (m3*bigm2logm1c1 + m4*bigm2logm2c2 + c3*bigm2 - bigm2y)

def gradient_c1():
    return 2 * (m3*bigc1logm1c1 + m4*bigc1logm2c2 + c3*bigc1 - bigc1y)

def gradient_c2():
    return 2 * (m3*bigc2logm1c1 + m4*bigc2logm2c2 + c3*bigc2 - bigc2y)

for _ in range (10000):

    logm1c1 = logm2c2 = logm1c12 = logm2c22 = logm1c1m2c2 = bigm1 = bigm2 = bigc1 = bigc2 = logm1c1y = logm2c2y = bigm1y = bigm2y = bigc1y = bigc2y = Y = bigm1logm1c1 = bigm2logm2c2 = bigm1logm2c2 = bigm2logm1c1 = bigc1logm1c1 = bigc1logm2c2 = bigc2logm1c1 = bigc2logm2c2 = 0

    for t in range(14):
        Y += rating[t]
        logm1c1 += soft1_fun(dosage[t])
        logm2c2 += soft2_fun(dosage[t])
        logm1c12 += soft1_fun(dosage[t]) ** 2
        logm2c22 += soft2_fun(dosage[t]) ** 2
        logm1c1y += soft1_fun(dosage[t]) * rating[t]
        logm2c2y += soft2_fun(dosage[t]) * rating[t]
        logm1c1m2c2 += soft1_fun(dosage[t]) * soft2_fun(dosage[t])
        bigm1 += bigm1_fun(dosage[t])
        bigm2 += bigm2_fun(dosage[t])
        bigc1 += bigc1_fun(dosage[t])
        bigc2 += bigc2_fun(dosage[t])
        bigm1y += bigm1_fun(dosage[t]) * rating[t]
        bigm2y += bigm2_fun(dosage[t]) * rating[t]
        bigc1y += bigc1_fun(dosage[t]) * rating[t]
        bigc2y += bigc2_fun(dosage[t]) * rating[t]
        bigm1logm1c1 += soft1_fun(dosage[t]) * bigm1_fun(dosage[t])
        bigm2logm2c2 += soft2_fun(dosage[t]) * bigm2_fun(dosage[t])
        bigm2logm1c1 += soft1_fun(dosage[t]) * bigm2_fun(dosage[t])
        bigm1logm2c2 += soft2_fun(dosage[t]) * bigm1_fun(dosage[t])
        bigc1logm1c1 += soft1_fun(dosage[t]) * bigc1_fun(dosage[t])
        bigc2logm2c2 += soft2_fun(dosage[t]) * bigc2_fun(dosage[t])
        bigc2logm1c1 += soft1_fun(dosage[t]) * bigc2_fun(dosage[t])
        bigc1logm2c2 += soft2_fun(dosage[t]) * bigc1_fun(dosage[t])


    soft1 = [soft1_fun(x) for x in dosage]
    soft2 = [soft2_fun(x) for x in dosage]
    summ = [sum_fun(x) for x in dosage]
    plt.scatter(dosage, rating)

    plt.plot(dosage, summ, color="black")
    plt.pause(0.0001)
    plt.clf()

    m1 = m1 - gradient_m1()*learning_rate
    m2 = m2 - gradient_m2()*learning_rate
    m3 = m3 - gradient_m3()*learning_rate
    m4 = m4 - gradient_m4()*learning_rate
    c1 = c1 - gradient_c1()*learning_rate
    c2 = c2 - gradient_c2()*learning_rate
    c3 = c3 - gradient_c3()*learning_rate



""" plt.scatter(dosage, soft1, color="red")
    plt.scatter(dosage, soft2, color="magenta")"""
