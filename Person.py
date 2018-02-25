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
		self.occlusion = None
		self.attention = None

		#not sure about these
		self.hands = None
		self.leftLooking = None
		self.rightLooking = None
		self.cropped = None


	def accumulateData():
		#probably need to be more selective and refined
		self.data = [self.blur, self.poseAngle, self.poseDistance, self.postureLR, self.occlusion, self.attention]


class LabelsForPerson:

	def __init__(self):
		#human labels
		self.labelIdentifier = ""
		self.humanFace = None # does participant think this is a face
		self.humanMovement = None # participant estimated movement (related to blur)
		self.humanPoseAngle = None # participant estimated angle
		self.humanEyeAngle = None # participant estimated eye gaze angle
		self.humanPostureLR = None # participant estimated left to right posture
		self.humanOcclusion = None # participant estimated occlusion
		self.humanAttention = None # participant estimated human attention
		#accumulated data
		self.data = [] #includes the 4 features of movement, eye angle, posture and occlusion
