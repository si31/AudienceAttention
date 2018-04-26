import sys
import cv2
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import HelperFunctions


def plotScatter(x, y):
	plt.plot(x, y, alpha=0.5)
	plt.title('Scatter plot')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()


def lineOfBestFit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b


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


def createGraph(xData, yData, title, xTitle, yTitle, labels=None):
	plt.plot(xData, yData, '-o', alpha=0.5, c=labels)
	#plt.plot(x,y, alpha = 0.5, color='black')
	plt.ylim([-1,1])
	plt.title(title)
	plt.xlabel(xTitle)
	plt.ylabel(yTitle)
	a,b = lineOfBestFit(xData, yData)
	yfit = [a + b * xi for xi in X]
	plt.plot(X, yfit)
	plt.show()