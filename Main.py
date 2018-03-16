import sys
import os
import cv2
import numpy as np
import uuid
import base64
import pickle

import FaceDetection
import ComputerVision
import MachineLearning
import HeadDirection
import PostureDetection
import HelperFunctions
import HAAR

from Person import Person, accumulateData, printData
from Image import Image
from Video import Video
import tkinter as tk
from PIL import ImageTk, Image as PILIm

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
		person.blur = ComputerVision.blur(person.image)
		pass
	print('Detecting occlusion, posture...')
	ComputerVision.findEars(img, mark=False)
	PostureDetection.getPosture(img)
	ComputerVision.finalMerge(img)
	#ComputerVision.findMovement(img)


def calculateAttention(img):
	#need to load model and then use it to predict attention for each person
	#need to pass image to the accumulate data function to allow it to get its blur for each section	
	if True:# if model does not exist
		print('Attention not estimated as model does not exist.')
	else:
		print('Estimating attention...')

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
			else:
				img = Image(imgFile)
				FaceDetection.findFaces(img, mark=False)
			detectFeatures(img)
		calculateAttention(img)
		#ComputerVision.findMovement(img)
		for person in img.persons:
			accumulateData(person)

	return img


USER_INPUT = None
root = None
imgGLO = None


def displayInterface(img):
	global USER_INPUT, root, imgGLO
	imgGLO = img
	root = tk.Tk()
	USER_INPUT = tk.StringVar(root)
	root.title("Audience Attention Analyser")
	root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
	root.focus_set()
	tk.Label(root, text="Overall Attention = ").pack()
	tk.Label(root, text="View person: ").pack()
	tk.Entry(root, textvariable=USER_INPUT).pack()
	tk.Button(root, text="Go", command=viewPerson).pack()
	tk.Button(root, text="Quit", command=root.destroy).pack()
	ratio = 1150/img.image.shape[1]
	height = int(ratio * img.image.shape[0])
	imgToShow = cv2.resize(img.image, (1150,height), interpolation=cv2.INTER_LINEAR);
	for index, person in enumerate(img.persons): #annotation always happens after saving so the image is not affected
		HelperFunctions.annotateImage(imgToShow, person.face, 42, index, ratio)
	imgToShow = cv2.cvtColor(imgToShow, cv2.COLOR_BGR2RGB)
	imgPIL = PILIm.fromarray(imgToShow)
	imgTk = ImageTk.PhotoImage(imgPIL)
	imgView = tk.Label(image=imgTk).pack(side=tk.BOTTOM)
	root.mainloop()


def viewPerson():
	global USER_INPUT, imgGLO
	index = int(USER_INPUT.get())
	person = imgGLO.persons[index]
	printData(person)


def handleImageFile(imgName):
	print('Reading image...')
	imgFile = cv2.imread('imgsInDatabase/'+imgName)
	img = handleImage(imgName, imgFile)

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

	#if sys.argv[4] == '1':
	#	displayInterface(img)

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


if __name__ == "__main__":
	main()
