import sklearn as skl
from sklearn.linear_model import LinearRegression

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def plotScatter(x, y):
	plt.scatter(x, y, alpha=0.5)
	plt.title('Scatter plot')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

def createLinearRegressionModel(x,y, plot=False):
	xArr = [[val] for val in x]

	model  = LinearRegression()
	model.fit(xArr, y)

	if plot:
		predicted_values = [model.coef_ * i + model.intercept_ for i in x]

		plt.scatter(x, y, alpha=0.5)
		plt.plot(x,predicted_values, alpha = 0.5, color='black')

		plt.title('Scatter plot')
		plt.xlabel('x')
		plt.ylabel('y')

		plt.show()

	return model

def linearRegressionPredict(model, val):
	return model.coef_ * val + model.intercept_

def createPolynomialRegressionModel(x,y,n,plot=False):
	
	poly = np.poly1d(np.polyfit(x,y,n))

	if plot:
		predicted_values = [poly(val) for val in x]

		plt.scatter(x, y, alpha=0.5)
		plt.plot(x,predicted_values, alpha = 0.5, color='black')

		plt.title('Scatter plot')
		plt.xlabel('x')
		plt.ylabel('y')

		plt.show()

	return model	
	pass

def learn(arrays):
	pass



def test(array):
	pass



