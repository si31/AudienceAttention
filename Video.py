import sys
import HelperFunctions 

class Video:

	def __init__(self):
		self.frames = []
		self.persons = [] #array of videPerson objects

	def trackPersonsAcrossFrames(self):
		for index in range(len(frames)-1):
			trackPersonsAcrossFrames(frames[i], frames[i+1])

	def trackPersonsAcrossTwoFrames(frame1, frame2):

class VideoPerson:

	def __init__(self):
		self.imagePersons = [] # length of no. of frames, each points to the person in the frame or None if no person detected
		self.averagePosition = (0,0,0,0)
		self.averageAttention = 0
