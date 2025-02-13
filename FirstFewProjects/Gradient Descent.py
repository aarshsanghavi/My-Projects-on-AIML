import matplotlib.pyplot as plt
import numpy as np
import math

x= np.arange(-2,2,0.1)
y= np.arange(-1.5,1.5,0.1)

X,Y= np.meshgrid(x,y)

def z_function(inpx,inpy):
    return np.sin(inpx*5)+np.cos(inpy*5)

def derivitive (inpx,inpy):
    return np.cos(inpx*5)*5 , -np.sin(inpy*5)*5

current_pos=(1,1,z_function(1,1))

fig = plt.figure(figsize =(14,9))
ax = plt.axes(projection ='3d' , computed_zorder=False)
learning_rate=0.001
for _ in range(1000):
    partx , party = derivitive(current_pos[0],current_pos[1])
    newx=current_pos[0]-(partx*learning_rate)
    newy=current_pos[1] - (party * learning_rate)
    current_pos=(newx,newy,z_function(newx,newy))
    ax.scatter(current_pos[0], current_pos[1], current_pos[2], color="red",zorder=1)
    ax.plot_surface(X, Y, z_function(X, Y), cmap="viridis" , zorder=0)
    plt.pause(0.001)
    ax.clear()


