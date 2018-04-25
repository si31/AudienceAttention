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
	plt.show()