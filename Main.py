import sys
import cv2
import numpy as np
import FaceDetection
import ComputerVision
import MachineLearning
import uuid
import base64


def saveToDatabase(img, imgName, persons):

	"""

	for saving with random file name

	unique_filename = str(uuid.uuid4())
	f = open('Database/' + unique_filename + ".txt", 'w')
	imgBytes = cv2.imencode('.jpg', img)[1]
	imgString = base64.b64encode(imgBytes)
	f.write(imgString.decode("utf-8"))
	"""
	f = open('Database/' + imgName + '.text', 'w')
	for person in persons:
		cv2.imshow('image1', person.image)
		(x,y,w,h) = person.face
		f.write(str(x) + '|' + str(y) + '|' + str(w) + '|' + str(h) + '|')
		key = cv2.waitKey(0)
		resultOfKey = ""
		if key == ord('y'):
			resultOfKey = "1"
		elif key == ord('n'):
			resultOfKey = "0"
		else:
			resultOfKey = "-1"
		f.write(resultOfKey + '|')
		landmarksToWrite = ''
		for (x,y) in person.landmarks:
			landmarksToWrite = landmarksToWrite + '|' + str(x) + ',' + str(y)
		f.write(landmarksToWrite)
		f.write('\n')
	f.close()


def inDatabase(img):
	return False


def loadFromDatabase(img):
	print('loading')


def saveImage(img):
	cv2.imwrite("/Users/admin/desktop/saved.jpg", img)


def showImage(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)


def runUntilBreak(fun, persons):
	for person in persons:
		fun(person, mark=False)


def main():
	print('Start')
	imageName = 'test_front2.jpg'
	img = cv2.imread(imageName)
	if inDatabase(img):
		print('in database')
	else:
		persons = FaceDetection.findFaces(img, mark=True)
		runUntilBreak(ComputerVision.faceLandmarks, persons)
		imgOriginal = cv2.imread(imageName)
		saveToDatabase(imgOriginal, imageName, persons)

	print('End')


if __name__ == "__main__":
	main()
