import sys
import cv2
import numpy as np
import EyeTracking
import HelperFunctions
import dlib
import pickle
import math

def edgeDetection(person):
	img = person.image
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(img,100,150)
	#cv2.imshow('title',edges)
	#cv2.imshow('title2',img)
	#keyPressed = cv2.waitKey()
	#return not keyPressed == 113
	return True

def getCharacteristics(img, persons):
	EyeTracking.findEyes(img, persons[0])


def showFaceImages(img, persons):
	for person in persons:
		croppedImage = person.image
		cv2.imshow('main',croppedImage)
		keyPressed = cv2.waitKey()
		if keyPressed == 113:
			break


def applyFilter(img, filterArray):
	newImage = np.zeros(img.shape)
	for x in range(0,img.shape[0]):
		for y in range(0,img.shape[1]):
			w = filterArray.shape[0]
			h = filterArray.shape[1]
			total = 0
			for fx in range(0,w):
				for fy in range(0,h):
					xOffset = x+(fx-(w//2))
					yOffset = y+(fy-(h//2)) #floor division
					if xOffset >= 0 and xOffset < img.shape[0] and yOffset >= 0 and yOffset < img.shape[1]:
						total += filterArray[fx][fy] * img[xOffset][yOffset]
			newImage[x][y] = total
	return newImage

def standardDeviationTwoImagesSingleChannel(img1, img2):
	total = 0
	for x in range(0,img1.shape[1]):
		for y in range(0,img1.shape[0]):
			total += math.pow(math.fabs(img1[x][y] - img2[x][y]),2)
	total = total / (img1.shape[1] + img1.shape[0])
	return total

def blur(image):
	lapacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	filtered = applyFilter(gray, lapacian)
	sd = standardDeviationTwoImagesSingleChannel(gray, filtered)
	return blur

def faceLandmarks(person, mark=False):
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	
	rect = HelperFunctions.dlibBBToRect(0,0,person.face[2],person.face[3])

	gray = cv2.cvtColor(person.image, cv2.COLOR_BGR2GRAY)

	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy array
	landmarks = predictor(gray, rect)
	landmarks = HelperFunctions.shape_to_np(landmarks)
	person.landmarks = landmarks.tolist()
	if mark:
		# loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
		for (x, y) in landmarks:
			cv2.circle(person.image, (x, y), 1, (0, 0, 255), -1)


def readFromDatabase(imgName):
	print('Reading from database...')
	with open(imgName + '.txt', 'rb') as f:
		person = pickle.load(f)
	return person

def main():
	blur(readFromDatabase('ArmTest/arm1').image)
if __name__ == "__main__":
	main()
