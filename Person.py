import sys
import cv2
import HelperFunctions 

class Person:

	def __init__(self, img, face, cascadeIdentifier):
		self.face = face
		self.cascadeIdentifier = cascadeIdentifier
		self.image = HelperFunctions.cropImage(img, self, 0)
		self.imageExtra = HelperFunctions.cropImageRatio(img, self, 0.25)
		self.landmarks = []
		self.humanAttention = None
		self.attention = None
		self.humanFace = None
		self.poseAngle = None
		self.poseDistance = None
		self.skin = None
		self.blur = None
		self.hands = None
		self.leftLooking = None
		self.rightLooking = None
		self.data = []
		self.cropped = None
		self.hogDrawing = None

	def accumulateData():
		print(poseAngle)
		print(poseDistance)
		self.data = [poseAngle, poseDistance, blur]
