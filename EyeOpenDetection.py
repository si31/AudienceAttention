import sys
import cv2
import numpy as np
import pickle

from Image import Image
from Person import Person

import HOG
import HelperFunctions


def removeDuplicates(detected):
	detectedNoDups = []
	for bboxA in detected:
		found = False
		for (bboxB, totalInGroup) in detectedNoDups:
			if HelperFunctions.bbOverLapRatio(bboxA,bboxB) > 0.01:
				bboxB = tuple([(totalInGroup*bboxB[index]+bboxA[index]) // (totalInGroup + 1) for index in range(0,4)]) #tuple([(len(bboxes)*bboxes[0][0][index]+bboxA[index]) // (len(bboxes) + 1) for index in range(0,len(bboxes[0][0]))])
				found = True
				break
		if not found:
			detectedNoDups.append((bboxA, 1))
	return detectedNoDups


def readFromDatabase(imgName):
	print('Reading from database...')
	with open('Database/' + imgName + '.txt', 'rb') as f:
		img = pickle.load(f)
	return img


def eyeExtractor():
	cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_eye.xml')
	img = readFromDatabase(sys.argv[1])
	for person in img.persons:
		detected = cascade.detectMultiScale(person.image, 1.1, 1)
		withoutDups = removeDuplicates(detected)
		cv2.imshow('img', person.image)
		cv2.waitKey(0)



def detectEyesOpen():
	img = cv2.imread('imgsInDatabase/'+sys.argv[1])

	cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_eye.xml')

	detected = cascade.detectMultiScale(img, 1.1, 1)

	hogView1 = None

	for face in detected:
		face = face.astype(int)
		face = face.tolist()
		(x,y,w,h) = face
		eye1 = img[y:y+h, x:x+w]
		hogView1 = HOG.getHOG(eye1)
		cv2.imshow('img1',eye1)
		cv2.imshow('img2',hogView1)
		cv2.waitKey(0)

		#cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)



	cv2.imshow('img', img)
	cv2.waitKey(0)




if __name__ == "__main__":
	#detectEyesOpen()
	eyeExtractor()