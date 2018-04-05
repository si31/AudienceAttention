import sys
import os
import cv2 

from Person import Person, LabelsForPerson, accumulateData, printData
from Image import Image
import HelperFunctions

import tkinter as tk
import pickle
from PIL import ImageTk, Image as PILIm

class Application(tk.Frame):
	def __init__(self, master=None):
		super().__init__(master)
		self.pack()
		self.create_widgets()

	def create_widgets(self):
		self.hi_there = tk.Button(self)
		self.hi_there["text"] = "Hello World\n(click me)"
		self.hi_there["command"] = self.say_hi
		self.hi_there.pack(side="top")

		self.quit = tk.Button(self, text="QUIT", fg="red", command=root.destroy)
		self.quit.pack(side="bottom")

	def say_hi(self):
		print("hi there, everyone!")


root = None


def createGUI():
	app = Application(master=root)
	app.mainloop()


def runImage():

	global img, phase, index
	person = img.persons[index]

	if len(person.labels) != 0:
		myLabel = person.labels[0]
		if myLabel.humanFace == 0:
			index += 1
			phase = 1
			if index > len(img.persons)-1:
				exit()
			runImage()
			return

	accumulateData(person)
	printData(person)
	labels = []
	for label in person.labels[1:]:
		labels.append(label.humanAttention)
	print([labels.count(2), labels.count(1), labels.count(0), labels.count(-1), labels.count(-2)])
	imgToShow = cv2.cvtColor(person.image, cv2.COLOR_BGR2RGB)
	imgPIL0 = PILIm.fromarray(imgToShow)
	imgTk0 = ImageTk.PhotoImage(imgPIL0)
	imgView0 = tk.Label(image=imgTk0)
	label0 = tk.Label(text=str(index))
	label0.pack(side=tk.LEFT)
	imgView0.pack(side=tk.LEFT)

	imgToShowExtra = cv2.cvtColor(person.imageExtra, cv2.COLOR_BGR2RGB)
	imgPIL = PILIm.fromarray(imgToShowExtra)
	imgTk = ImageTk.PhotoImage(imgPIL)
	imgView = tk.Label(image=imgTk)
	imgView.pack(side=tk.LEFT)

	elementsToPack = []
	
	if phase == 0:
		#buttons to select face
		b1 = tk.Button(text="Is a face", command=lambda: nextImage(1))
		b0 = tk.Button(text="Is not a face", command=lambda: nextImage(0))
		elementsToPack = [b1, b0]
	elif phase == 1:
		#buttons to select attention
		l0 = tk.Label(text="How attentive is this person on a scale from 1 to 5?")
		b0 = tk.Button(text="5", command=lambda: nextImage(2))
		b1 = tk.Button(text="4", command=lambda: nextImage(1))
		b2 = tk.Button(text="3", command=lambda: nextImage(0))
		b3 = tk.Button(text="2", command=lambda: nextImage(-1))
		b4 = tk.Button(text="1", command=lambda: nextImage(-2))
		elementsToPack = [l0, b0, b1, b2, b3, b4]
	elif phase == 2:
		#buttons to select headpose angle
		bUL = tk.Button(text="upper left", command=lambda: nextImage(1))
		bUM = tk.Button(text="upper middle", command=lambda: nextImage(2))
		bUR = tk.Button(text="upper right", command=lambda: nextImage(3))
		bML = tk.Button(text="middle left", command=lambda: nextImage(4))
		bMM = tk.Button(text="middle middle", command=lambda: nextImage(5))
		bMR = tk.Button(text="middle right", command=lambda: nextImage(6))
		bLL = tk.Button(text="lower left", command=lambda: nextImage(7))
		bLM = tk.Button(text="lower middle", command=lambda: nextImage(8))
		bLR = tk.Button(text="lower right", command=lambda: nextImage(9))
		elementsToPack = [bUL, bUM, bUR, bML, bMM, bMR, bLL, bLM, bLR]
	elif phase == 3:
		#buttons to select eyegaze angle
		bUL = tk.Button(text="upper left", command=lambda: nextImage(1))
		bUM = tk.Button(text="upper middle", command=lambda: nextImage(2))
		bUR = tk.Button(text="upper right", command=lambda: nextImage(3))
		bML = tk.Button(text="middle left", command=lambda: nextImage(4))
		bMM = tk.Button(text="middle middle", command=lambda: nextImage(5))
		bMR = tk.Button(text="middle right", command=lambda: nextImage(6))
		bLL = tk.Button(text="lower left", command=lambda: nextImage(7))
		bLM = tk.Button(text="lower middle", command=lambda: nextImage(8))
		bLR = tk.Button(text="lower right", command=lambda: nextImage(9))
		elementsToPack = [bUL, bUM, bUR, bML, bMM, bMR, bLL, bLM, bLR]
	elif phase == 4:
		#buttons to select movement
		b1 = tk.Button(text="Person is moving", command=lambda: nextImage(1))
		b0 = tk.Button(text="Person is not moving", command=lambda: nextImage(0))
		elementsToPack = [b1, b0]
	elif phase == 5:
		#buttons to select occlusion
		b1 = tk.Button(text="Occluded", command=lambda: nextImage(1))
		b0 = tk.Button(text="Not occluded", command=lambda: nextImage(0))
		elementsToPack = [b1, b0]
	elif phase == 6:
		#buttons to select postureLR
		b2 = tk.Button(text="Left posture", command=lambda: nextImage(2))
		b1 = tk.Button(text="Centre posture", command=lambda: nextImage(1))
		b0 = tk.Button(text="Right posture", command=lambda: nextImage(0))
		elementsToPack = [b2, b1, b0]
	for element in elementsToPack:
		element.pack(side=tk.TOP)
	root.mainloop()


def getLabelObject(person):
	identifiers = {label.labelIdentifier:label for label in person.labels}
	if labelIdentifier in identifiers:
		print('found')
		return identifiers[labelIdentifier]
	else:
		newLabel = LabelsForPerson()
		newLabel.labelIdentifier = labelIdentifier
		person.labels.append(newLabel)
		return newLabel


def nextImage(val):
	MAX_PHASE = 1
	global img, phase, labelsForPerson, index, root, objToSave
	person = img.persons[index]
	print(index)
	if phase == 0:
		labelsForPerson = getLabelObject(person)
		labelsForPerson.humanFace = val
		if val == 0:
			phase = MAX_PHASE
	elif phase == 1:
		labelsForPerson = getLabelObject(person)
		labelsForPerson.humanAttention = val
		print('run')
	elif phase == 2:
		labelsForPerson.humanPoseAngle = val
	elif phase == 3:
		labelsForPerson.humanEyeAngle = val
	elif phase == 4:
		labelsForPerson.humanMovement = val
	elif phase == 5:
		labelsForPerson.humanOcclusion = val
	elif phase == 6:
		labelsForPerson.humanPostureLR = val
	if phase == MAX_PHASE:
		index += 1
		#labelsForPerson.data = [labelsForPerson.humanEyeAngle, labelsForPerson.humanMovement, labelsForPerson.humanOcclusion, labelsForPerson.humanPostureLR]
		print(img.persons[0].labels)
		HelperFunctions.saveToDatabase(objToSave, sys.argv[1])
		if index > len(img.persons)-1:
			exit()
		phase = 1
	else:
		phase += 1
	root.destroy()
	root = tk.Tk()
	root.title("Audience Attention Labeller")
	root.geometry('1500x800+100+100')
	runImage() 


img = None
index = 0
phase = 1
labelsForPerson = None
labelIdentifier = ""
objToSave = None


def mainVideo():
	global img, labelIdentifier, root, index, objToSave

	# looking through the video, participant looks at a person throughout and then gives a score of their attention overall. do for each person in the mode frame

	print('Usage: vidName, command, label identifier, startFrom, frame')
	video = HelperFunctions.readFromDatabase(sys.argv[1])
	command = sys.argv[2]

	img = video.frameWithMostDetections()

	objToSave = video

	print(len(img.persons))

	if command == "ls":
		examplePerson = img.persons[0]
		identifiers = [label.labelIdentifier for label in examplePerson.labels]
		if len(identifiers) == 0:
			print('no identifiers found')
		else:
			print('--')
			for identifier in identifiers:
				print(identifier)
			print('--')

	elif command == "clear-make-sure":
		for person in img.persons:
			person.labels = []
		HelperFunctions.saveToDatabase(objToSave, sys.argv[1])

	elif command == "new":
		labelIdentifier = sys.argv[3]
		root = tk.Tk()  
		root.title("Audience Attention Labeller")
		index = int(sys.argv[4])
		runImage()

	elif command == "delete":	
		labelIdentifier = sys.argv[3]
		for person in img.persons:
			for label in person.labels:
				if label.labelIdentifier == labelIdentifier:
					person.labels.remove(label)
		HelperFunctions.saveToDatabase(img, sys.argv[1])

	print('End')


def main():
	
	global img, labelIdentifier, root, index, objToSave

	print('Usage: imgName, command, label identifier, startFrom')
	img = HelperFunctions.readFromDatabase(sys.argv[1])
	command = sys.argv[2]

	print(len(img.persons))

	objToSave = img

	if command == "ls":
		examplePerson = img.persons[0]
		identifiers = [label.labelIdentifier for label in examplePerson.labels]
		if len(identifiers) == 0:
			print('no identifiers found')
		else:
			print('--')
			for identifier in identifiers:
				print(identifier)
			print('--')

	elif command == "clear-make-sure":
		for person in img.persons:
			person.labels = []
		HelperFunctions.saveToDatabase(img, sys.argv[1])

	elif command == "new":
		labelIdentifier = sys.argv[3]
		root = tk.Tk()  
		root.title("Audience Attention Labeller")
		index = int(sys.argv[4])
		runImage()

	elif command == "delete":	
		labelIdentifier = sys.argv[3]
		for person in img.persons:
			for label in person.labels:
				if label.labelIdentifier == labelIdentifier:
					person.labels.remove(label)
		HelperFunctions.saveToDatabase(img, sys.argv[1])

	print('End')
	return


if __name__ == "__main__":
	main()