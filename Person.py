import sys
import cv2
import HelperFunctions 

class Person:

	def __init__(self, img, face, cascadeIdentifier):

		#information
		self.face = face
		self.cascadeIdentifier = cascadeIdentifier
		self.landmarks = []
		self.data = []
		self.labels = []

		#images
		self.image = HelperFunctions.cropImage(img, self, 0)
		self.imageExtra = HelperFunctions.cropImageRatio(img, self, 2)
		self.skin = None
		self.hogDrawing = None
	
		#detected attributes
		self.blur = None
		self.poseAngle = None
		self.poseDistance = None
		self.postureLR = None
		self.postureFB = None
		self.attention = None

		#not sure about these
		self.hands = None
		self.leftLooking = None
		self.rightLooking = None
		self.cropped = None


	def accumulateData():
		print(poseAngle)
		print(poseDistance)
		self.data = [poseAngle, poseDistance, blur]

class LabelsForPerson:

	def __init__(self):
		#human labels
		self.labelIdentifier = ""
		self.humanFace = None # does participant think this is a face
		self.humanMovement = None # participant estimated movement (related to blur)
		self.humanPoseAngle = None # participant estimated angle
		self.humanEyeAngle = None # participant estimated eye gaze angle
		self.humanPostureLR = None # participant estimated left to right posture
		self.humanPostureFB = None # participant estimated forward to back posture
		self.humanAttention = None # participant estimated human attention
