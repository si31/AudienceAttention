import sys
import cv2
import numpy as np
import HelperFunctions
from Person import Person
import MachineLearning
import math 

def convertToObjects(img, withoutDups):
	persons = []
	for face in withoutDups:
		persons.append(Person(img.image, face[0],face[1])) #could add a better way to choose the best face
	return persons


def removeDuplicates(detected):
	detectedNoDups = []
	for (bboxA, cascadeIdentifierA) in detected:
		found = False
		for (bboxB,cascadeIdentifierB, totalInGroup) in detectedNoDups:
			if HelperFunctions.bbOverLapRatio(bboxA,bboxB) > 0.01:
				bboxB = tuple([(totalInGroup*bboxB[index]+bboxA[index]) // (totalInGroup + 1) for index in range(0,4)]) #tuple([(len(bboxes)*bboxes[0][0][index]+bboxA[index]) // (len(bboxes) + 1) for index in range(0,len(bboxes[0][0]))])
				found = True
				break
		if not found:
			detectedNoDups.append((bboxA, cascadeIdentifierA, 1))
	return detectedNoDups


def getCascades():
	cascadePaths = ['/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml',
					'/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_profileface.xml',
					'/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml']
	cascades = []
	for cascadePath in cascadePaths:
		cascades.append(cv2.CascadeClassifier(cascadePath))
	return cascades

def splitImageVertically(img, n):
	n += 1
	interval = img.shape[1]//n
	interval = interval//2
	sections = []
	for i in range(0,n*2):
		lowerVal = i*interval
		upperVal = lowerVal+interval
		if upperVal >= img.shape[1]:
			upperVal = img.shape[1]
		sections.append((img[lowerVal:upperVal, 0:img.shape[0]], lowerVal))
	return sections[:-1]

def removeUnlikelyFaces(detected, minFactor, maxFactor):
	print('Removing unlikely faces...')
	datapoints = []
	for face in detected:
		((x,y,w,h),cascadeIdentifier, totalInGroup) = face
		datapoints.append((y, math.sqrt(w*h)))
	x = [i[0] for i in datapoints]
	y = [i[1] for i in datapoints]
	linearRegressionModel = MachineLearning.createLinearRegressionModel(x,y,plot=False)
	numRemoved = 0
	for face in detected:
		((x,y,w,h), cascadeIdentifier, totalInGroup) = face
		predictedArea = MachineLearning.linearRegressionPredict(linearRegressionModel, y)
		if (math.sqrt(w*h) < predictedArea*minFactor or math.sqrt(w*h) > predictedArea * maxFactor):
			numRemoved += 1
			detected.remove(face)
	if numRemoved > len(detected)*0.025:
		detected = removeUnlikelyFaces(detected, minFactor*1.1 if minFactor < 0.7 else minFactor, maxFactor*0.9 if maxFactor > 1.75 else maxFactor)

	return detected

def findFaces(img, mark=False):
	print('Detecting faces...')
	cascades = getCascades()
	detectedAll = []
	totalSize = 0 
	withoutDups = []

	#sections = splitImageVertically(img.image, 4) #tried to split up but didnt work

	#for (section, lowerVal) in sections:
	for index, cascade in enumerate(cascades):
		cascadeIdentifier = [0] * len(cascades)
		cascadeIdentifier[index] = 1
		for val in [p/10 for p in range(12, 13)]:
			detected = cascade.detectMultiScale(img.image, val, 1)
			#detectedFlipped = cascade.detectMultiScale(cv2.flip(img.image, 1), 1.2, 1)
			for face in detected:
				face = face.astype(int)
				face = face.tolist()
				detectedAll.append((face, cascadeIdentifier))
			"""
		for face in detectedFlipped:
			face = face.astype(int)
			face = face.tolist()
			face[0] = img.image.shape[0] - face[0] - face[2]
			#detectedAll.append((face, cascadeIdentifier))
			"""

	withoutDups = removeDuplicates(detectedAll)

	"""for items in withoutDups:
		overallCascadeIdentifier = [0] * len(cascades)
		for item in items:
			overallCascadeIdentifier = [x + y for (x, y) in zip(overallCascadeIdentifier, item[1])] #combines the cascade identifiers so that it has data of each cascade that found it
		items[0] = (items[0][0], overallCascadeIdentifier) #gives the first face (atm the one that gets used) the overall cascade identifier
	
	withoutDupsSingleFace = []

	for face in withoutDups:
		withoutDupsSingleFace.append(items[0]) #could add a better way to choose the best box - biggest or average them
	"""
	withoutDups = removeUnlikelyFaces(withoutDups, 0.5, 2.5)

	if mark:
		print('mark')
		#just for visual markers of each face
		for item in withoutDups:
			((x,y,w,h),cascadeIdentifier,totalInGroup) = item 
			cv2.rectangle(img.image,(x,y),(x+w,y+h),(0,0,255),2)

	img.persons = convertToObjects(img, withoutDups)
