import sys
import HelperFunctions 

class Video:

	def __init__(self):
		self.frameImages = [] # actual images
		self.frames = [] #image objects
		self.persons = [] #array of videoPerson objects

	def collatePersons(self):
		for person in frames[0].persons:
			videoPerson = VideoPerson(person)
			self.persons.append(videoPerson, 0)
		for index, frame in enumerate(self.frames[1:]):
			for personInFrame in frame2.persons:
				trackPersonsAcrossFrames(index, personInFrame, nextFrame)

	def trackPersonAcrossTwoFrames(index, personInFrame, nextFrame): #each are Image objects
		found = False
		for videoPerson in self.persons:
			if HelperFunctions.bbOverLapRatio(personInFrame.face, videoPerson.averagePosition) > 0.1:
				found = True
				videoPerson.addFrame(personInFrame, index)
				break
		if not found:
			self.persons.append(VideoPerson(personInFrame, index))

	def fillInFrameHoles():
		pass # fill in frames that do not have a person in from not being detected


class VideoPerson:

	def __init__(self, initalFramePerson, initalFrameIndex):
		self.imagePersons = [None] * len(self.frames) # length of no. of frames, each points to the person in the frame or None if no person detected
		self.imagePersons[initalFrameIndex] = initalFramePerson 
		self.averagePosition = (0,0,0,0)
		self.averageAttention = 0
		self.numberOfFramesIn = 1

	def addFrame(self, personInFrame, frameIndex):
		self.imagePersons[frameIndex] = personInFrame
		self.numberOfFramesIn += 1
		self.averagePosition = self.newAveragePosition(personInFrame.face)

	def calcNewAveragePosition(self, newPosition):
		(x1,y1,w1,h1) = self.averagePosition
		(x2,y2,w2,h2) = self.face
		self.averagePosition = ((x1+x2)//2,(y1+y2)//2,(w1+w2)//2,(h1+h2)//2)

	def calcAverageAttention(self):
		attentionSum = 0
		for imagePerson in self.imagePersons:
			if imagePerson is not None:
				attentionSum += imagePerson.attention
		self.averagePosition = attentionSum/self.numberOfFramesIn

