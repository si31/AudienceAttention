import sys
import HelperFunctions 

class Video:

	def __init__(self):
		self.frameImages = [] # actual images
		self.frames = [] #image objects
		self.persons = [] #array of videoPerson objects
		self.attention = 0.0

	def collatePersons(self): #run command here
		print('Collating...')
		self.persons = []
		for person in self.frames[0].persons:
			videoPerson = VideoPerson(person, 0, self)
			self.persons.append(videoPerson)
		for index, frame in enumerate(self.frames[1:]):
			for personInFrame in frame.persons:
				self.trackPersonAcrossFrames(index, personInFrame)
		#remove ones with just one detection
		toRemove = []
		for videoPerson in self.persons:
			if videoPerson.numberOfFramesIn <= 2:
				toRemove.append(videoPerson)
		for bad in toRemove:
			self.persons.remove(bad)
		for person in self.persons:
			person.calcAverageAttention()

	def trackPersonAcrossFrames(self, index, personInFrame): #personinframe is an Image object
		found = False
		for videoPerson in self.persons:
			if HelperFunctions.bbOverLapRatio(personInFrame.face, videoPerson.averagePosition) > 0.1:
				found = True
				videoPerson.addFrame(personInFrame, index)
				break
		if not found:
			self.persons.append(VideoPerson(personInFrame, index, self))

	def calculateOverallAttention(self):
		for videoPerson in self.persons:
			self.attention += videoPerson.attention
		self.attention = self.attention / len(self.persons)

	def frameWithMostDetections(self):
		highestNumber = 0
		frameWithHighest = 0
		for frame in self.frames:
			if len(frame.persons) > highestNumber:
				highestNumber = len(frame.persons)
				frameWithHighest = frame
		return frameWithHighest


class VideoPerson:

	def __init__(self, initalFramePerson, initalFrameIndex, video): #initial frame index is the index of the frame they were first seen in
		self.imagePersons = [None] * len(video.frames) # length of no. of frames, each points to the person in the frame or None if no person detected, image objects
		self.imagePersons[initalFrameIndex] = initalFramePerson 
		self.averagePosition = initalFramePerson.face
		self.attention = 0
		self.numberOfFramesIn = 1

	def addFrame(self, personInFrame, frameIndex):
		print(frameIndex)
		self.imagePersons[frameIndex+1] = personInFrame
		self.numberOfFramesIn += 1
		self.averagePosition = personInFrame.face#self.calcNewAveragePosition(personInFrame.face)

	def calcNewAveragePosition(self, newPosition):
		(x1,y1,w1,h1) = self.averagePosition
		(x2,y2,w2,h2) = newPosition
		self.averagePosition = ((x1+x2)//2,(y1+y2)//2,(w1+w2)//2,(h1+h2)//2)

	def calcAverageAttention(self):
		attentionSum = 0.0
		print(self.imagePersons)
		for imagePerson in self.imagePersons:
			if imagePerson is not None:
				attentionSum += imagePerson.attention
				print(imagePerson.attention)
		print(self.numberOfFramesIn)
		self.attention = attentionSum/self.numberOfFramesIn
