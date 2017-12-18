import sys
import cv2
import HelperFunctions 

class Person:

	def __init__(self, img, face, cascadeIdentifier):
		self.face = face
		self.cascadeIdentifier = cascadeIdentifier
		self.image = HelperFunctions.cropImage(img, self, 0)
		self.imageExtra = HelperFunctions.cropImage(img, self, 100)
		self.landmarks = []
		self.humanAttention = None
		self.attention = None
		self.humanFace = None
