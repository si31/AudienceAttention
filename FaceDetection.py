import sys
import cv2
import numpy as np
import HelperFunctions
from Person import Person

def convertToObjects(img, withoutDups):
	persons = []
	for faces in withoutDups:
		persons.append(Person(img.img, faces[0][0],faces[0][1])) #could add a better way to choose the best face
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


def findFaces(img, mark=False):
	print('Detecting faces...')
	cascades = getCascades()
	detectedAll = []
	totalSize = 0 
	for index, cascade in enumerate(cascades):
		detected = cascade.detectMultiScale(img.img, 1.2, 1)
		cascadeIdentifier = [0] * len(cascades)
		cascadeIdentifier[index] = 1
		for face in detected:
			face = face.astype(int)
			face = face.tolist()
			detectedAll.append((face, cascadeIdentifier))
	withoutDups = removeDuplicates(detectedAll)

	for items in withoutDups:
		overallCascadeIdentifier = [0] * len(cascades)
		for item in items:
			overallCascadeIdentifier = [x + y for (x, y) in zip(overallCascadeIdentifier, item[1])] #combines the cascade identifiers so that it has data of each cascade that found it
		items[0] = (items[0][0], overallCascadeIdentifier) #gives the first face (atm the one that gets used) the overall cascade identifier
	
	if mark:
		#just for visual markers of each face
		for items in withoutDups:
			((x,y,w,h),cascadeIdentifier) = items[0] #could add a better way to choose the best box - biggest or average them
			cv2.rectangle(img.img,(x,y),(x+w,y+h),(0,0,255),2)

	img.persons = convertToObjects(img, withoutDups)
