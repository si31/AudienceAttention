import sys
import os
import cv2
import numpy as np
import uuid
import base64
import pickle
import tkinter as tk
from PIL import ImageTk, Image as PILIm

import FaceDetection
import ComputerVision
import MachineLearning
import HeadDirection
import PostureDetection
import HelperFunctions
import HAAR
import GraphCreator
import Teach
import sklearn

from Person import Person, accumulateData, printData
from Image import Image
from Video import Video

def showImage(img):
	print('Showing image...')
	img = cv2.resize(img, (1280, 960))  
	cv2.imshow('image',img)
	cv2.waitKey(0)


def showAllPeople(persons):
	print('Showing all people...')
	for person in persons:
		cv2.imshow('image',	person.image)
		accumulateData(person)
		printData(person)
		key = cv2.waitKey(0)
		if key == ord('q'):
			break
		elif key == ord('s'):
			HelperFunctions.saveObject(person)
		elif key == ord('m'):
			HelperFunctions.saveImage(person.image)


def detectFeatures(img):
	print('Detecting landmarks, head pose, blur...')
	for person in img.persons:
		ComputerVision.faceLandmarks(person, mark=False)
		HeadDirection.getPose(person, img.image.shape, mark=False)
		person.blur = 0 #ComputerVision.blur(person.image)
		pass
	print('Detecting occlusion, posture...')
	#ComputerVision.findEars(img, mark=False)
	PostureDetection.getPosture(img)
	FaceDetection.removeUnlikelyFacesFinal(img.persons, 0.25, 2.25)
	ComputerVision.finalMerge(img)
	#ComputerVision.findMovement(img)


def calculateAttention(img):
	#need to load model and then use it to predict attention for each person
	exists = False
	for file in os.listdir():
		if file == 'model.pkl':
			exists = True
			break
	if exists:
		print('Estimating attention...')
		classifier = None
		with open('model.pkl', 'rb') as f:
			classifier = pickle.load(f)
		averageAttention = 0.0
		for person in img.persons:
			result = Teach.svmTestSK(classifier, [person.data])
			[attention] = result.tolist()
			person.attention = attention/2
			averageAttention += person.attention
		img.attention = averageAttention / len(img.persons)
	else:
		print('Attention not estimated as model does not exist.')


def handleImage(imgName, imgFile):
	img = None

	if HelperFunctions.inDatabase(imgName) and sys.argv[2] == '3':
		img = HelperFunctions.readFromDatabase(imgName)
	else:
		if HelperFunctions.inDatabase(imgName) and sys.argv[2] == '2':
			img = HelperFunctions.readFromDatabase(imgName)
		else:
			if HelperFunctions.inDatabase(imgName) and sys.argv[2] == '1':
				img = HelperFunctions.readFromDatabase(imgName)
				affect(img)
			else:
				img = Image(imgFile)
				FaceDetection.findFaces(img, mark=False)
			detectFeatures(img)
		#ComputerVision.findMovement(img)
		for person in img.persons:
			accumulateData(person)
		calculateAttention(img)

	return img


USER_INPUT = None
root = None
imgGLO = None
videoGLO = None


def affect(img):
	image = img.image
	k = -30
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			image[x][y] = np.array([x+k if k+x > 0 else 0 for x in image[x][y].tolist()])
	#cv2.imshow('img', img.image)
	#cv2.waitKey(0)


def displayInterface(img, video=False):
	global USER_INPUT, root, imgGLO
	imgGLO = img
	root = tk.Tk()
	USER_INPUT = tk.StringVar(root)
	root.title("Audience Attention Analyser")
	root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
	root.focus_set()
	tk.Label(root, text="Overall Attention = " + str(img.attention)).pack()
	tk.Label(root, text="View person: ").pack()
	tk.Entry(root, textvariable=USER_INPUT).pack()
	tk.Button(root, text="Print details of person", command=viewPerson).pack()
	if video:
		tk.Button(root, text="View attention graph of person", command=viewGraphOfPerson).pack()
		tk.Button(root, text="View attention graph of group", command=viewGraphOfAttention).pack()
	tk.Button(root, text="Quit", command=root.destroy).pack()
	ratio = 1150/img.image.shape[1]
	height = int(ratio * img.image.shape[0])
	imgToShow = cv2.resize(img.image, (1150,height), interpolation=cv2.INTER_LINEAR);
	for index, person in enumerate(img.persons): #annotation always happens after saving so the image is not affected
		HelperFunctions.annotateImage(imgToShow, person.face, '{0:.0f}'.format(person.attention*100), index, ratio)
	imgToShow = cv2.cvtColor(imgToShow, cv2.COLOR_BGR2RGB)
	imgPIL = PILIm.fromarray(imgToShow)
	imgTk = ImageTk.PhotoImage(imgPIL)
	imgView = tk.Label(image=imgTk).pack(side=tk.BOTTOM)
	root.mainloop()


def viewGraphOfPerson():
	global USER_INPUT, imgGLO, videoGLO
	video = videoGLO
	index = int(USER_INPUT.get())
	person = imgGLO.persons[index]
	videoPersonSelected = None
	for videoPerson in video.persons:
		print('1')
		print(videoPerson.imagePersons)
		if person in videoPerson.imagePersons:
			print('Found person')
			videoPersonSelected = videoPerson
	#display graph
	data = [(val.attention, index) for index, val in enumerate(videoPersonSelected.imagePersons) if val is not None]
	print(data)
	xData = [a for a,b in data]
	yData = [b for a,b in data]
	GraphCreator.createGraph(xData, yData, 'hi', 'bye', 'poo')


def viewGraphOfAttention():
	global USER_INPUT, imgGLO
	index = int(USER_INPUT.get())
	person = imgGLO.persons[index]
	printData(person)
	pass


def viewPerson():
	global USER_INPUT, imgGLO
	index = int(USER_INPUT.get())
	person = imgGLO.persons[index]
	printData(person)


def handleImageFile(imgName):
	print('Reading image...')
	imgFile = cv2.imread('imgsInDatabase/'+imgName)

	if imgFile is None:
		print('File could not be read')
		return

	img = handleImage(imgName, imgFile)

	global evalValue
	evalValue = Teach.main()

	if sys.argv[3] == '1':
		HelperFunctions.saveToDatabase(img, imgName)

	if sys.argv[4] == '1':
		displayInterface(img)


def handleVideoFile(vidName, frameInterval):
	print('Reading video...')
	if HelperFunctions.inDatabase(vidName) and (sys.argv[2] == '2' or sys.argv[2] == '3'):
		print('Using saved version...')
		video = HelperFunctions.readFromDatabase(vidName)
		if sys.argv[2] == '2':
			video.collatePersons()
	else:
		print('Recalculating...')
		file = cv2.VideoCapture('imgsInDatabase/'+vidName)
		video = Video()
		i = 0

		while(file.isOpened()):
			ret, frame = file.read()
			if ret and frame is not None:
				i += 1
				if i % frameInterval != 0:
					continue
				video.frameImages.append(frame)
				video.frames.append(handleImage(vidName, frame))
				print('Frame ' + str(i//frameInterval) + ' completed.')
			else:
				break
		file.release()
		cv2.destroyAllWindows()
 
		video.collatePersons()

	for person in video.persons:    
		print(person.imagePersons)

	if sys.argv[3] == '1':
		HelperFunctions.saveToDatabase(video, vidName)

	if sys.argv[4] == '1':
		global videoGLO
		videoGLO = video
		displayInterface(video.frameWithMostDetections(), True)

	#run viewer on frame with mode people in, plus give option of graphs and each person's graph


def main():
	print('-----------------------------')
	print('Usage: file name, use saved if available (3=used saved everything, 2=used saved redo attention, 1=used saved faces, 0=redo everything), save to database, view faces')
	print('For Video: (3=used saved everything, 2=redo the collation, 1,0=redo everything')
	print('Start...')
	
	if len(sys.argv) < 5:
		print('Not enough parameters specified')
		return

	fileName = sys.argv[1]
	if fileName.endswith('.jpg') or fileName.endswith('.png'):
		handleImageFile(fileName)
	elif fileName.endswith('.mp4'):
		handleVideoFile(fileName, 15)
	else:
		print('Input file is of wrong type. Please use .jpg or .png for images and .mp4 for videos.')

	print('End')


evalValue = []


def autoMain():
	fileNames = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4a.jpg', 'img4b.jpg',
				'img4c.jpg', 'img6.jpg','img7.jpg', 'img8.jpg', 'img9a.jpg',
				'img9b.jpg', 'img9c.jpg', 'img10.jpg']
	global evalValue
	for fileName in fileNames:
		handleImageFile(fileName)
	[postureCorrect, postureIncorrect, occlusionCorrect, occlusionIncorrect, postureOcclusionNA, poseCorrect, poseIncorrect, correctDetections] = evalValue
	print(evalValue)


if __name__ == "__main__":
	autoMain()
