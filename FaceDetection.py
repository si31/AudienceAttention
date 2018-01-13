import sys
import cv2
import numpy as np
import HelperFunctions
from Person import Person
import MachineLearning
import math 

def convertToObjects(img, withoutDups):
	persons = []
	for faces in withoutDups:
		persons.append(Person(img.image, faces[0][0],faces[0][1])) #could add a better way to choose the best face
	return persons


def removeDuplicates(detected):
	detectedNoDups = []
	for (bboxA, cascadeIdentifier) in detected:
		found = False
		for bboxes in detectedNoDups:
			bboxB = bboxes[0][0]
			if HelperFunctions.bbOverLapRatio(bboxA,bboxB) > 0.5:
				bboxes.append((bboxA, cascadeIdentifier))
				found = True
				break
		if not found:
			detectedNoDups.append([(bboxA, cascadeIdentifier)])
	return detectedNoDups


def getCascades():
	cascadePaths = ['/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml',
					'/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_profileface.xml',
					'/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml']#,
					#'/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml',
					#'/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml']
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

def removeUnlikelyFaces(detected):
	datapoints = []
	for face in detected:
		((x,y,w,h),cascadeIdentifier) = face
		datapoints.append((y, math.sqrt(w*h)))
	x = [i[0] for i in datapoints]
	y = [i[1] for i in datapoints]
	linearRegressionModel = MachineLearning.createLinearRegressionModel(x,y,plot=False)
	numRemoved = 0
	for face in detected:
		((x,y,w,h), cascadeIdentifier) = face
		predictedArea = MachineLearning.linearRegressionPredict(linearRegressionModel, y)
		if (math.sqrt(w*h) < predictedArea*0.75 or math.sqrt(w*h) > predictedArea * 2.5):
			numRemoved += 1
			detected.remove(face)
	if numRemoved > len(detected)*0.025:
		detected = removeUnlikelyFaces(detected)

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
		detected = cascade.detectMultiScale(img.image, 1.2, 1)
		cascadeIdentifier = [0] * len(cascades)
		cascadeIdentifier[index] = 1
		for face in detected:
			face = face.astype(int)
			face = face.tolist()
			#face[1] += lowerVal
			detectedAll.append((face, cascadeIdentifier))
	
	withoutDups = removeDuplicates(detectedAll)

	for items in withoutDups:
		overallCascadeIdentifier = [0] * len(cascades)
		for item in items:
			overallCascadeIdentifier = [x + y for (x, y) in zip(overallCascadeIdentifier, item[1])] #combines the cascade identifiers so that it has data of each cascade that found it
		items[0] = (items[0][0], overallCascadeIdentifier) #gives the first face (atm the one that gets used) the overall cascade identifier

	withoutDupsSingleFace = []

	for items in withoutDups:
		withoutDupsSingleFace.append(items[0]) #could add a better way to choose the best box - biggest or average them

	withoutDupsSingleFace = removeUnlikelyFaces(withoutDupsSingleFace)

	if mark:
		print('mark')
		#just for visual markers of each face
		for item in withoutDupsSingleFace:
			((x,y,w,h),cascadeIdentifier) = item 
			cv2.rectangle(img.image,(x,y),(x+w,y+h),(0,0,255),2)

	img.persons = convertToObjects(img, withoutDups)
