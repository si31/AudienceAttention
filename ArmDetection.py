import sys
import cv2
import numpy as np
import HelperFunctions
from Person import Person
import ComputerVision
import dlib
import math
import pickle
from scipy.stats import itemfreq

#from tutorial
def getArms(img, mark=False):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	wide = cv2.Canny(blurred, 10, 200)
	tight = cv2.Canny(blurred, 225, 250)
	# compute the median of the single channel pixel intensities
	v = np.median(blurred)
	sigma = 0.33
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(tight, lower, upper)
 
	cv2.imshow('name', img)
	cv2.waitKey(0)
	cv2.imshow('name', edged)
	cv2.waitKey(0)

	pass

def sittingBodyShapeOnly(person):
	img = person.imageExtra
	face = person.face
	shape = img.shape
	(xmax, ymax) = shape[0:2]
	x1 = xmax/2 - face[3]/2
	y1 = ymax/2
	k1 = (ymax-y1) / math.sqrt(xmax-x1)
	for x in range(0, shape[0]):
		for y in range(0, shape[1]):
			#f(x) = +-k . sqrt(x-x1) + y1
			if (1 >= 0):
				#f1 = k1 * math.sqrt(x-x1) + y1
				#f2 = -k1 * math.sqrt(x-x1) + y1
				f1 = +math.pow((y-y1)/k1,2) + x1
				f2 = -math.pow((y-y1)/k1,2) + x1
				"""
				colour line
				if (f1 > y-1 and f1 < y+1) or (f2 > y-1 and f2 < y+1):
					print('draw')
					img[x,y] = [255,192,203]
				"""
				if (x < f1):
					img[x,y] = [0,0,0]					


	#cv2.imshow('name', img)
	#cv2.waitKey(0)

#copied from stack overflow
def dominant_color(img):
	
	arr = np.float32(img)
	pixels = arr.reshape((-1, 3))

	n_colors = 2
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	_, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

	palette = np.uint8(centroids)
	quantized = palette[labels.flatten()]
	quantized = quantized.reshape(img.shape)

	dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]

	return dominant_color


#from tutorial
def detectSkin(person):
	img = person.imageExtra

	faceColour = dominant_color(cv2.cvtColor(person.image, cv2.COLOR_BGR2HSV))
	lowerColourList = [x-25 for x in faceColour]
	upperColourList = [x+75 for x in faceColour]
	lower = np.array(lowerColourList)
	upper = np.array(upperColourList)

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(hsv, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
 
	# blur the mask to help remove noise, then apply the
	# mask to the frame
	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	skin = cv2.bitwise_and(img, img, mask = skinMask)
 
	# show the skin in the image along with the mask
	#cv2.imshow("name", np.hstack([img, skin]))
 
	#cv2.waitKey(0)
	return skin

def detectSkin2(person):
	img = person.image
	chromed = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	mask = np.zeros(img.shape)

	lower = np.array([25, 25, 25])
	uppper = np.array([200,200,200])

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(hsv, lower, upper)

	skin = np.zeros(img.shape)
	img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	skin = cv2.bitwise_and(img, img, mask = mask)

	cv2.imshow('name', np.hstack([img, chromed, mask, skin]))
	cv2.waitKey(0)
	return mask

def readFromDatabase(imgName):
	print('Reading from database...')
	with open(imgName + '.txt', 'rb') as f:
		person = pickle.load(f)
	return person


def getSkin(person):
	#sittingBodyShapeOnly(person)
	skin = detectSkin(person)
	person.skin = skin
	#getArms(skin)

