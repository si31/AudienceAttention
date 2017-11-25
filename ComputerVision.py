import sys
import cv2
import numpy as np
import EyeTracking
import HelperFunctions
import dlib


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


def faceLandmarks(person, mark=False):
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	
	image = person.image
	rect = HelperFunctions.dlibBBToRect(0,0,person.face[2],person.face[3])

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	landmarks = predictor(gray, rect)
	landmarks = HelperFunctions.shape_to_np(landmarks)
	person.landmarks = landmarks.tolist()
	if mark:
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		cv2.imshow('original',image)
		for (x, y) in landmarks:
			print(x)
			print(y)
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	
	
	#cv2.imshow('landmarks',image)
	#keyPressed = cv2.waitKey()
	return True #not keyPressed == 113