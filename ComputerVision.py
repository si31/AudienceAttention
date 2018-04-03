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
	newImage = img.copy()
	for x in range(0,img.shape[0]):
		for y in range(0,img.shape[1]):
			w = filterArray.shape[0]
			h = filterArray.shape[1]
			total = 0.0
			for fx in range(0,w):
				for fy in range(0,h):
					xOffset = x+(fx-(w//2))
					yOffset = y+(fy-(h//2))
					if xOffset >= 0 and xOffset < img.shape[0] and yOffset >= 0 and yOffset < img.shape[1]:
						total += filterArray[fx][fy] * img[xOffset][yOffset]
			total = total if total < 256 else 255
			newImage[x][y] = total if total >= 0 else 0
	return newImage


def varianceTwoImagesSingleChannel(img1):
	totalSquares = 0
	total = 0
	overallTotal = 0
	overallTotalSquares = 0
	for x in range(img1.shape[0]):
		for y in range(img1.shape[1]):
			total += (img1[x][y])
		overallTotal += total / (img1.shape[0] * img1.shape[1])
		total = 0
	mean = overallTotal
	for x in range(0,img1.shape[0]):
		for y in range(0,img1.shape[1]):
			totalSquares += math.pow((img1[x][y]) - mean, 2)
		overallTotalSquares += totalSquares / (img1.shape[0] * img1.shape[1])
		totalSquares = 0
	var = overallTotalSquares
	return var


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
	image = cv2.resize(image, (300,300), interpolation=cv2.INTER_LINEAR);
	lapacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	filtered = applyFilter(gray, lapacian)
	sd = varianceTwoImagesSingleChannel(filtered)
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

	mac = True
	if mac:
		cascadePaths.append('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_mcs_rightear.xml')
		cascadePaths.append('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_mcs_leftear.xml')
	else:
		cascadePaths.append('/home/simon/opencv/data/haarcascades/haarcascade_mcs_rightear.xml')
		cascadePaths.append('/home/simon/opencv/data/haarcascades/haarcascade_mcs_leftear.xml')

	for cascadePath in cascadePaths:
		cascade = cv2.CascadeClassifier(cascadePath)
		result = cascade.detectMultiScale(img.image, 1.05, 1)
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


def finalMerge(img):
	newPersons = []
	for personA in img.persons:
		found = False
		for personB in newPersons:
			if HelperFunctions.bbOverLapRatio(personA.face, personB.face) > 0.5:
				personB.face = [(x+y)//2 for x,y in list(zip(personA.face, personB.face))]
				print(personB.face)
				found = True
				break
		if not found:
			newPersons.append(personA)
	img.persons = newPersons


def readFromDatabase(imgName):
	print('Reading from database...')
	with open(imgName + '.txt', 'rb') as f:
		person = pickle.load(f)
	return person


def main():
	image = cv2.imread('imgsInDatabase/me1.png')
	sd = blur(image)
	print(sd)
	print('--')
	image = cv2.imread('imgsInDatabase/me2.png')
	sd = blur(image)
	print(sd)
	print('--')
	image = cv2.imread('imgsInDatabase/me3.png')
	sd = blur(image)
	print(sd)
	print('--')
	image = cv2.imread('imgsInDatabase/me4.png')
	sd = blur(image)
	print(sd)

if __name__ == "__main__":
	main()
