import sys
sys.path.append('/home/simon/PyOpenPose/build/PyOpenPoseLib')
import os

import PyOpenPose as OP
import time
import cv2
import numpy as np
import math

from Image import Image
import HelperFunctions

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
	#res = op.render(img) #get OP to render the body positions onto the image
	dataOP = op.getKeypoints(op.KeypointType.POSE)[0]
	dataOP = dataOP.tolist()
	faces = op.faceRects.tolist()
	hands = op.handRects.tolist()
	personOPPointss = []
	for index, person in enumerate(dataOP):
		keypoints = PersonOPPoints(faces[index], hands[index], person)
		personOPPointss.append(keypoints)
	
	view = False;
	if view:
		#just for visual markers of each face
		for person in personOPPointss:
			(x,y,w,h) = person.rightHandBB
			cv2.rectangle(res,(x,y),(x+w,y+h),(0,0,255),4)
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
		self.leftKnee = keypoints[9] #might be foot
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


def associatePersons(personsA, personsB):
	associations = []
	for personA in personsA:
		for personB in personsB:
			if HelperFunctions.bbOverLapRatio(personA.face, personB.face) > 0.01:
				associations.append((personA, personB))
				break
	return associations


def determineHandPositionType(face, leftHand, rightHand):
	leftOcclusion = HelperFunctions.bbOverLapRatio(face, leftHand) > 0.01
	rightOcclusion = HelperFunctions.bbOverLapRatio(face, rightHand) > 0.01
	occlusion = leftOcclusion or rightOcclusion
	return (occlusion)


def getPosture(img):
	detectedPersons = img.persons #what has been detected by program so far
	detectedPersonsOP = runOP(img.image)
	associations = associatePersons(detectedPersons, detectedPersonsOP)
	for (personA, personB) in associations:
		(occlusion) = determineHandPositionType(personB.face, personB.leftHandBB, personB.rightHandBB)
		personA.occlusion = occlusion
		personA.postureLR = shoulderAngle(personB.leftShoulder, personB.rightShoulder)


if __name__ == '__main__':
	runOP(Image(cv2.imread('imgsInDatabase/img1.jpg')).image)
