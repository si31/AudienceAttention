import sys
import cv2
import numpy as np
import FaceDetection
import ComputerVision
import MachineLearning
import HeadDirection
import ArmDetection
import uuid
import base64
from Person import Person
from Image import Image
import pickle

def saveToDatabase(img, imgName):
	print('Saving to database...')
	with open('Database/' + imgName + '.txt', 'wb') as f:
		pickle.dump(img, f)

def label(img):
	for person in img.persons:
		cv2.imshow('image1', person.image)
		key = cv2.waitKey(0)
		resultOfKey = ""
		if key == ord('y'):
			person.attention = True
			person.humanFace = True
		elif key == ord('n'):
			person.attention = False
			person.humanFace = True
		else:
			person.humanFace = False

def readFromDatabase(imgName):
	print('Reading from database...')
	with open('Database/' + imgName + '.txt', 'rb') as f:
		img = pickle.load(f)
	return img

def inDatabase(imgName):
	fileNames = fileToArray('Database/filenames.txt')
	return ((imgName + '.txt') in fileNames)

def fileToArray(imgName):
	with open(imgName) as f:
		content = f.readlines()
		content = [x.strip() for x in content]
	return content

def saveImage(img):
	cv2.imwrite("/Users/admin/desktop/saved.jpg", img)

def saveObject(obj):
	print('Saving obj to desktop...')
	with open("/Users/admin/desktop/saved.txt", 'wb') as f:
		pickle.dump(obj, f)

def showImage(img):
	print('Showing image...')
	cv2.imshow('image',img)
	cv2.waitKey(0)

def showAllPeople(persons):
	print('Showing all people...')
	for person in persons:
		print(person.image)
		for x in range(0, person.image.shape[0]):
			print(person.image[x])
		cv2.imshow('image',	person.image)
		key = cv2.waitKey(0)
		if key == ord('q'):
			break
		elif key == ord('s'):
			saveObject(person)

def main():
	print('image name, use saved if available, create labels, save to database, view faces')	
	print('Start...')

	imgName = sys.argv[1]
	imgFile = cv2.imread('imgsInDatabase/'+imgName)
	img = None
	if inDatabase(imgName) and sys.argv[2] == 'y':
		img = readFromDatabase(imgName)
	else:
		img = Image(imgFile)
		FaceDetection.findFaces(img, mark=True)
		print('Detecting landmarks, pose, skin, blur...')
		for person in img.persons:
			ComputerVision.faceLandmarks(person, mark=False)
			HeadDirection.getPose(person, img.image.shape, mark=False)
			person.blur = ComputerVision.blur(person.image)
			ArmDetection.getSkin(person)
		img.blur = ComputerVision.blur(img.image)
		saveToDatabase(img, imgName)

	if sys.argv[3] == 'y':
		label(img)

	if sys.argv[4] == 'y':
		saveToDatabase(img, imgName)

	if sys.argv[5] == 'y':
		showImage(img.image)
		cv2.waitKey(0)
		showAllPeople(img.persons)

	print('End')


if __name__ == "__main__":
	main()
