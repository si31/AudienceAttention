import sys
import os

from Person import Person, LabelsForPerson 
from Image import Image

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


def saveToDatabase(img, imgName):
	print('Saving to database...')
	with open('Database/' + imgName + '.txt', 'wb') as f:
		pickle.dump(img, f)


def readFromDatabase(imgName):
	print('Reading from database...')
	with open('Database/' + imgName + '.txt', 'rb') as f:
		img = pickle.load(f)
	return img


def runImage():

	global img, phase
	person = img.persons[index]

	imgPIL = PILIm.fromarray(person.imageExtra)
	imgTk = ImageTk.PhotoImage(imgPIL)
	imgView = tk.Label(image=imgTk)  
	imgView.pack()

	elementsToPack = []
	
	if phase == 0:
		#buttons to select face
		b1 = tk.Button(text="Is a face", command=lambda: nextImage(1))
		b0 = tk.Button(text="Is not a face", command=lambda: nextImage(0))
		elementsToPack = [b1, b0]
	elif phase == 1:
		#buttons to select attention
		b1 = tk.Button(text="Shows attention", command=lambda: nextImage(1))
		b0 = tk.Button(text="Does not show attention", command=lambda: nextImage(0))
		elementsToPack = [b1, b0]
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
		#buttons to select face
		b1 = tk.Button(text="Is a face", command=lambda: nextImage(1))
		b0 = tk.Button(text="Is not a face", command=lambda: nextImage(0))
		elementsToPack = [b1, b0]
	for element in elementsToPack:
		element.pack()
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

	global img, phase, labelsForPerson, index, root
	person = img.persons[index]

	if phase == 0:
		labelsForPerson = getLabelObject(person)
		labelsForPerson.humanFace = val
		if val == 0:
			index += 1
			phase = 0
	elif phase == 1:
		labelsForPerson.humanAttention = val
	elif phase == 2:
		labelsForPerson.humanPoseAngle = val
	elif phase == 3:
		labelsForPerson.humanEyeAngle = val
	elif phase == 4:
		labelsForPerson.humanMovement = val

	if phase == 4: # max phase number
		index += 1
		phase = 0
	else:
		phase += 1
	root.destroy()
	root = tk.Tk()
	root.title("Audience Attention Labeller") 
	runImage() 


img = None
index = 0
phase = 0
labelsForPerson = None
labelIdentifier = ""

def main():
	
	global img, labelIdentifier, root
	img = readFromDatabase(sys.argv[1])
	labelIdentifier = sys.argv[2]

	root = tk.Tk()  
	root.title("Audience Attention Labeller")

	runImage()

	print('End')
	return


if __name__ == "__main__":
	main()