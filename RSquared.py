from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import math
import random

style.use('fivethirtyeight')

#xs = np.array([2,4,6,7,8,9,4,5], dtype = np.float64)
#ys = np.array([9,7,3,4,5,2,4,5], dtype = np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
	val = 1
	ys = []
	for i in range(hm):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step

	xs = [i for i in range(len(ys))]
	return np.array(xs, dtype = np.float64), np.array(ys, dtype= np.float64)

def slope_intercept(xs,ys):
	m = ((mean(xs) * mean(ys)) - mean(xs*ys)) / ((math.pow(mean(xs), 2)) - mean(xs*xs))
	b = ( (mean(ys)) - (m * mean(xs)) )
	return m,b

def squared_error(ys_point, ys_reg):
	return sum ((ys_reg - ys_point)**2)

def coeffOfDetermination(ys_point,ys_reg):
	y_mean_line = [mean(ys_point) for y in ys_point]
	squared_error_reg = squared_error(ys_point, ys_reg)
	squared_error_y_Mean = squared_error(ys_point, y_mean_line)
	return 1 - (squared_error_reg / squared_error_y_Mean)

xs,ys = create_dataset(40, 60, 2, correlation = "pos")

m,b = slope_intercept(xs,ys)
regression_line = [(m*x) + b for x in xs]

r_squared = coeffOfDetermination(ys, regression_line)
print (r_squared)

#Predict values
predicted_x = 2
predicted_y = (m*predicted_x) + b

plt.scatter(xs,ys)
plt.scatter(predicted_x,predicted_y, s= 100, color='g')
plt.plot(xs,regression_line)
plt.show()
