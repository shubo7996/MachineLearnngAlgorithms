from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import math

style.use('fivethirtyeight')

xs = np.array([2,4,6,7,8,9,4,5], dtype = np.float64)
ys = np.array([9,7,3,4,5,2,4,5], dtype = np.float64)

def slope_intercept(xs,ys):
	m = ((mean(xs) * mean(ys)) - mean(xs*ys)) / ((math.pow(mean(xs), 2)) - mean(xs*xs))
	b = ( (mean(ys)) - (m * mean(xs)) )
	return m,b

m,b = slope_intercept(xs,ys)
regression_line = [(m*x) + b for x in xs]

#Predict values
predicted_x = 2
predicted_y = (m*predicted_x) + b

plt.scatter(xs,ys)
plt.scatter(predicted_x,predicted_y,color='g')
plt.plot(xs,regression_line)
plt.show()
