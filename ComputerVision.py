import sys
import cv2
import numpy as np
import EyeTracking
import HelperFunctions
import dlib
import pickle
import math
import MachineLearning


def edgeDetection(img):
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(img,50,150)
	return edges


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


def varianceTwoImagesSingleChannel(img1, img2):
	totalSquares = 0
	total = 0
	for x in range(0,img1.shape[0]):
		for y in range(0,img1.shape[1]):
			totalSquares += math.pow((img1[x][y] - img2[x][y]),2)
			total += (img1[x][y] - img2[x][y])
	squaresAverage = totalSquares / (img1.shape[0] + img1.shape[1])
	average = total / (img1.shape[0] + img1.shape[1])
	sd = squaresAverage - math.pow(average,2)
	return sd


def findMovement(img):
	print('Finding Movement...')
	datapoints = []
	for person in img.persons:
		(x,y,w,h) = person.face
		datapoints.append((y,person.blur))
	x = [i[0] for i in datapoints]
	y = [i[1] for i in datapoints]
	linearRegressionModel = MachineLearning.createLinearRegressionModel(x,y,plot=False)
	numRemoved = 0
	for person in img.persons:
		(x,y,w,h) = person.face
		predictedBlur = MachineLearning.linearRegressionPredict(linearRegressionModel, y)
		print('Predicted: ' + str(predictedBlur[0]))
		print('Actual: ' + str(person.blur))
		print('Ratio: ' + str(predictedBlur[0]/person.blur))
		cv2.imshow('img', person.image)
		cv2.waitKey(0)


def blur(image):
	lapacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	filtered = applyFilter(gray, lapacian)
	sd = varianceTwoImagesSingleChannel(gray, filtered)
	return sd


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
		for (x, y) in landmarks:
			cv2.circle(person.image, (x, y), 1, (0, 0, 255), -1)


def findEars(img, mark=False):
	cascadePaths = []
	detected = []

	mac = False
	if mac:
		cascadePaths.append('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_mcs_rightear.xml')
		cascadePaths.append('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_mcs_leftear.xml')
	else:
		cascadePaths.append('/home/simon/opencv/data/haarcascades/haarcascade_mcs_rightear.xml')
		cascadePaths.append('/home/simon/opencv/data/haarcascades/haarcascade_mcs_leftear.xml')

	for cascadePath in cascadePaths:
		cascade = cv2.CascadeClassifier(cascadePath)
		result = cascade.detectMultiScale(img.image, 1.05, 1)
		print(result)
		if result == ():
			continue
		detected += result.tolist()
	
	if mark:
		for item in detected:
			(x,y,w,h) = item 
			cv2.rectangle(img.image,(x,y),(x+w,y+h),(0,0,255),2)

	for person in img.persons:
		for ear in detected:
			if HelperFunctions.bbOverLapRatio(person.face, ear) != 0:
				person.earDetected = True
				break


def readFromDatabase(imgName):
	print('Reading from database...')
	with open(imgName + '.txt', 'rb') as f:
		person = pickle.load(f)
	return person


def main():
	blur(readFromDatabase('ArmTest/arm1').image)


if __name__ == "__main__":
	main()
