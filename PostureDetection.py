import sys
sys.path.append('/home/simon/PyOpenPose/build/PyOpenPoseLib')
import os

import PyOpenPose as OP
import time
import cv2
import numpy as np
import math

from Image import Image
from Person import Person
import HelperFunctions
import ComputerVision

OPENPOSE_ROOT = "/home/simon/openpose"


def runOP(img):

	height, width = img.shape[0:2]

	renderHeight = 480
	renderWidth = 640
	
	op = OP.OpenPose((320, 240), (240, 240), (renderWidth, renderHeight), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0, False)
 
	op.detectPose(img)
	op.detectFace(img)
	op.detectHands(img)
	res = np.copy(img)
	dataOP = op.getKeypoints(op.KeypointType.POSE)[0]
	dataOP = dataOP.tolist()
	dataOPFace = op.getKeypoints(op.KeypointType.POSE)[0].tolist()[0]

	faces = op.faceRects.tolist()
	hands = op.handRects.tolist()
	personOPPointss = []
	for index, person in enumerate(dataOP):
		keypoints = PersonOPPoints(faces[index], hands[index], person)
		personOPPointss.append(keypoints)
	
	view = False

	if view:
		res = op.render(img) #get OP to render the body positions onto the image
		res = cv2.resize(res, (640,480))
		cv2.imshow("OpenPose result", res)			
		key = cv2.waitKey(0)

	return personOPPointss
	

def shoulderAngle(leftShoulder, rightShoulder):
	if leftShoulder == [0,0,0] or rightShoulder == [0,0,0]:
		return 0
	x1, y1 = leftShoulder[0:2]
	x2, y2 = rightShoulder[0:2]
	if x1 == x2:
		print('x pos of shoulders the same: probable error')
		return 0
	return math.atan((y2-y1)/(x2-x1))


class PersonOPPoints:

	def __init__(self, face, hands, keypoints):
		self.face = face
		self.rightHandBB = self.resizeHand(face, hands[0:4])
		self.leftHandBB = self.resizeHand(face, hands[4:8])
		self.centreFace = keypoints[0] #maybe nose
		self.centreBody = keypoints[1] #middle of chest
		self.leftShoulder = keypoints[2]
		self.leftElbow = keypoints[3]
		self.leftHand = keypoints[4]
		self.rightShoulder = keypoints[5]
		self.rightElbow = keypoints[6]
		self.rightHand = keypoints[7]
		self.bottomLeftBody = keypoints[8]
		self.leftKnee = keypoints[9] #cant see
		self.leftFoot = keypoints[10] #cant see
		self.bottomRightBody = keypoints[11]
		self.rightKnee = keypoints[12]
		self.rightFoot = keypoints[13] #cant see
		self.leftEye = keypoints[14]
		self.rightEye = keypoints[15]
		self.leftSideOfHead = keypoints[16]
		self.rightSideOfHead = keypoints[17]

	def resizeHand(self, face, hand):
		(x1,y1,w1,h1) = hand
		handToFaceSizeRatio = 0.6 
		w2 = face[2]*handToFaceSizeRatio
		h2 = face[3]*handToFaceSizeRatio
		x2 = (w1-w2)/2 + x1
		y2 = (h1-h2)/2 + y1
		return (int(x2),int(y2),int(w2),int(h2))

	def calcHeadPose(self):
		maxDistance = self.face[2]/4
		xOfBody = self.centreBody[0]
		xOfFace = self.centreFace[0]
		print('next')
		side = (math.fabs(xOfFace-xOfBody) > maxDistance)
		if xOfFace == 0 or xOfBody == 0:
			side = False
		leftLower = (self.leftSideOfHead[1] - self.leftEye[1] < maxDistance/2)
		if self.leftSideOfHead[1] == 0 or self.leftEye[1] == 0:
			leftLower = False
		rightLower = (self.rightSideOfHead[1] - self.rightEye[1] < maxDistance/2)
		if self.rightSideOfHead[1] == 0 or self.leftEye[1] == 0:
			rightLower = False
		print(side)

		faceToBody = HelperFunctions.calcDistance(self.centreFace[0:2], self.centreBody[0:2])
		print(faceToBody)
		print(maxDistance*2)
		if side or faceToBody < maxDistance*1.75:#leftLower or rightLower:
			print('no')
			return 0
		else:
			print('yes')
			return 1
		#OPHead = (HelperFunctions.calcDistance(self.leftEye[0:2], self.leftSideOfHead[0:2]) < maxDistance and HelperFunctions.calcDistance(self.rightEye[0:2], self.rightSideOfHead[0:2]) < maxDistance)


def associatePersons(personsA, personsB):
	global imgGLO
	associations = []
	for personB in personsB:
		if personB.face != [0,0,0,0]:
			found = False
			for personA in personsA:
				if HelperFunctions.bbOverLapRatio(personA.face, personB.face) > 0.1:
					associations.append((personA, personB))
					personA.face = personB.face
					found = True
					break
			if not found:
				newPerson = Person(imgGLO.image, personB.face, None)
				ComputerVision.faceLandmarks(newPerson, mark=False)
				#HeadDirection.getPose(newPerson, imgGLO.image.shape, mark=False) # can get rid of as will overwrite it
				newPerson.blur = ComputerVision.blur(newPerson.image)
				imgGLO.persons.append(newPerson)
				newPerson.poseDistance = 1000
				associations.append((newPerson, personB))
	return associations


def determineHandPositionType(face, leftHand, rightHand):
	leftOcclusion = HelperFunctions.bbOverLapRatio(face, leftHand) > 0.01
	rightOcclusion = HelperFunctions.bbOverLapRatio(face, rightHand) > 0.01
	occlusion = 1 if leftOcclusion or rightOcclusion else 0
	return (occlusion)


imgGLO = None


def getPosture(img):
	print('start posture')
	global imgGLO
	imgGLO = img
	detectedPersons = img.persons #what has been detected by program so far
	detectedPersonsOP = runOP(img.image)
	associations = associatePersons(detectedPersons, detectedPersonsOP)
	for (personA, personB) in associations:
		(occlusion) = determineHandPositionType(personB.face, personB.leftHandBB, personB.rightHandBB)
		personA.occlusion = occlusion
		personA.headPoseOP = True
		personA.postureLR = shoulderAngle(personB.leftShoulder, personB.rightShoulder)
		personA.lookingForward = personB.calcHeadPose()


if __name__ == '__main__':
	runOP(Image(cv2.imread('imgsInDatabase/img1.jpg')).image)
