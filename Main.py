import sys
import os
import cv2
import numpy as np
import uuid
import base64
import pickle
import HOG
import time

import FaceDetection
import ComputerVision
import MachineLearning
import HeadDirection
import PostureDetection
import HAAR

from Person import Person
from Image import Image
from Video import Video


def saveToDatabase(img, imgName):
	print('Saving to database...')
	with open('Database/' + imgName + '.txt', 'wb') as f:
		pickle.dump(img, f)


def readFromDatabase(imgName):
	print('Reading from database...')
	with open('Database/' + imgName + '.txt', 'rb') as f:
		img = pickle.load(f)
	return img


def inDatabase(imgName):
	for file in os.listdir("Database/"):
		if file == imgName + '.txt':
			return True
	return False


def fileToArray(imgName):
	with open(imgName) as f:
		content = f.readlines()
		content = [x.strip() for x in content]
	return content


def saveImage(img):
	print('Saving img to desktop...')
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
		person.accumulateData()
		print(person.data)
		print(person.landmarks)
		print(person.image.shape)
		key = cv2.waitKey(0)
		if key == ord('q'):
			break
		elif key == ord('s'):
			saveObject(person)
		elif key == ord('m'):
			saveImage(person.image)


def detectFeatures(img):
	print('Detecting landmarks, head pose, blur...')
	for person in img.persons:
		ComputerVision.faceLandmarks(person, mark=False)
		HeadDirection.getPose(person, img.image.shape, mark=False)
		person.blur = ComputerVision.blur(person.image)
	print('Detecting occlusion, posture...')
	#PostureDetection.getPosture(img)
	#print('Detecting image blur...')
	#img.blur = ComputerVision.blur(img.image)


def calculateAttention(persons):
	#need to load model and then use it to predict attention for each person
	#need to pass image to the accumulate data function to allow it to get its blur for each section	
	if True:# if model does not exist
		print('Attention not estimated as model does not exist.')
	print('Estimating attention...')

def handleImage(imgName, imgFile=None):

	if imgFile is None:
		print('Reading image...')
		imgFile = cv2.imread('imgsInDatabase/'+imgName)
	
	img = None
	if inDatabase(imgName) and sys.argv[2] == '3':
		img = readFromDatabase(imgName)
	else:
		if inDatabase(imgName) and sys.argv[2] == '2':
			img = readFromDatabase(imgName)
		else:
			if inDatabase(imgName) and sys.argv[2] == '1':
				img = readFromDatabase(imgName)
			else:
				img = Image(imgFile)
				FaceDetection.findFaces(img, mark=False)
			detectFeatures(img)
		calculateAttention(img.persons)
		#ComputerVision.findMovement(img)
		for person in img.persons:
			person.accumulateData()

	if sys.argv[3] == '1':
		saveToDatabase(img, imgName)

	if sys.argv[4] == '1':
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
	print('Usage: file name, use saved if available (3=used saved everything, 2=used saved redo attention, 1=used saved faces, 0=redo everything), save to database, view faces')	
	print('Start...')
	
	if len(sys.argv) < 5:
		print('Not enough parameters specified')
		return

	fileName = sys.argv[1]
	if fileName.endswith('.jpg') or fileName.endswith('.png'):
		handleImage(fileName)
	elif fileName.endswith('.mp4'):
		handleVideo(fileName, 10)
	else:
		print('Input file is of wrong type. Please use .jpg or .png for images and .mp4 for videos.')

if __name__ == "__main__":
	main()
