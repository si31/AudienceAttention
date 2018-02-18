import sys
sys.path.append('/home/simon/PyOpenPose/build/PyOpenPoseLib')
import os

import PyOpenPose as OP
import time
import cv2
import numpy as np
import math

from Image import Image

OPENPOSE_ROOT = "/home/simon/openpose"

def runOP(img):

	height, width = img.shape[0:2]

	renderHeight = 480
	renderWidth = 640
	
	op = OP.OpenPose((320, 240), (240, 240), (renderWidth, renderHeight), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0, False)
 
	op.detectPose(img)
	op.detectFace(img)
	op.detectHands(img)

	res = op.render(img)
	persons = op.getKeypoints(op.KeypointType.POSE)[0]

	persons = persons.tolist()
	for person in persons:
		keypoints = PersonKeypoints(person)
		print(shoulderAngle(keypoints.leftShoulder, keypoints.rightShoulder))
		
	
	cv2.imshow("OpenPose result", res)			
	key = cv2.waitKey(0)

def shoulderAngle(leftShoulder, rightShoulder):
	if leftShoulder == [0,0,0] or rightShoulder == [0,0,0]:
		return 0
	x1, y1 = leftShoulder[0:2]
	x2, y2 = rightShoulder[0:2]
	if x1 == x2:
		print('x pos the same. probable error')
		return 0
	return math.atan((y2-y1)/(x2-x1))

class PersonKeypoints:

	def __init__(self, keypoints):
		self.face = None

		#pose points
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

def getPosture(img):
	runOP(img.image)
	detectedPersons = img.persons
	detectedPersonsOP = 

if __name__ == '__main__':
	getPosture(Image(cv2.imread('imgsInDatabase/img1.jpg')))

