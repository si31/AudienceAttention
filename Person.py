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
		self.image = HelperFunctions.cropImageRatio(img, self, 0.1)
		self.imageExtra = HelperFunctions.cropImageRatio(img, self, 2)
		self.skin = None
		self.hogDrawing = None
	
		#detected attributes
		self.blur = None
		self.poseAngle = None
		self.poseDistance = None
		self.postureLR = -1.0
		self.occlusion = -1
		self.attention = None

		#extra
		self.poseArea = None
		self.postureArea = None

		#not sure about these
		self.leftLooking = None
		self.rightLooking = None
		self.cropped = None


	def accumulateData(self):
		#probably need to be more selective and refined
		print(self.face)
		faceWidth = self.face[2]
		angle = self.poseAngle - 3.14/8
		if angle < -3.14:
			angle = angle + 6.28
		if self.poseDistance < faceWidth*0.5:
			self.poseArea = 5
		else:
			if angle < -2.36:
				self.poseArea = 4
			elif angle < -1.57:
				self.poseArea = 1
			elif angle < -0.79:
				self.poseArea = 2
			elif angle < 0:
				self.poseArea = 3
			elif angle < 0.79:
				self.poseArea = 6
			elif angle < 1.57:
				self.poseArea = 9
			elif angle < 2.36:
				self.poseArea = 8
			else:
				self.poseArea = 7
		if self.postureLR < -15:
			self.postureArea = 0
		elif self.postureLR > 15:
			self.postureArea = 2
		else:
			self.postureArea = 1
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
