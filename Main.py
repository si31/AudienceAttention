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
from Video import Video
import pickle
import HOG

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
	img = cv2.resize(img, (1280, 960))  
	cv2.imshow('image',img)
	cv2.waitKey(0)

def showAllPeople(persons):
	print('Showing all people...')
	for person in persons:
		cv2.imshow('image',	person.image)
		cv2.imshow('hog', person.hogDrawing)
		key = cv2.waitKey(0)
		if key == ord('q'):
			break
		elif key == ord('s'):
			saveObject(person)
		elif key == ord('m'):
			saveImage(person.image)
		elif key == ord('n'):
			saveImage('')


def handleImage(imgName, imgFile=None):
	if imgFile is None:
		print('Reading image...')
		imgFile = cv2.imread('imgsInDatabase/'+imgName)
	
	img = None

	if inDatabase(imgName) and sys.argv[2] == 'y':
		img = readFromDatabase(imgName)
	else:
		img = Image(imgFile)
		FaceDetection.findFaces(img, mark=False)
		print('Detecting landmarks, pose, skin, blur...')
		for person in img.persons:
			ComputerVision.faceLandmarks(person, mark=False)
			HeadDirection.getPose(person, img.image.shape, mark=True)
			hogDrawing = HOG.getHOG(person.image)
			person.hogDrawing = hogDrawing
			#person.blur = ComputerVision.blur(person.image)
			#hands = ArmDetection.getSkin(person)
		print('Detecting image blur...')
		#img.blur = ComputerVision.blur(img.image)
	if sys.argv[6] == 'y':
		print('Predicting attention...')
		for person in img.persons:
			person.accumulateData()
			#need to load model and then use it to predict attention for each person
			#need to pass image to the accumulate data function to allow it to get its blur for each section	

	if sys.argv[3] == 'y':
		label(img)

	if sys.argv[4] == 'y':
		saveToDatabase(img, imgName)

	if sys.argv[5] == 'y':
		showImage(img.image)
		showAllPeople(img.persons)

	print('End')
	return img

def handleVideo(vidName, frameInterval):
	print('Reading video...')
	cap = cv2.VideoCapture('imgsInDatabase/'+vidName)

	video = Video()
	i = 0

	while(cap.isOpened()):
		ret, frame = cap.read()
		i += 1

		if i % frameInterval != 0:
			continue

		video.frames.append(frame)
		video.frameObjects.append(handleImage(vidName, frame))

	cap.release()
	cv2.destroyAllWindows()

def main():
	print('file name, use saved if available, create labels, save to database, view faces, calculate attention')	
	print('Start...')

	fileName = sys.argv[1]

	if fileName.endswith('.jpg'):
		handleImage(fileName)
	elif fileName.endswith('.mp4'):
		handleVideo(fileName, 10)
	else:
		print('Input file is of wrong type. Please use .jpg for images and .mp4 for videos.')
	
if __name__ == "__main__":
	main()
