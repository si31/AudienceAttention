import sys
import cv2
import HelperFunctions

class Person:

	def __init__(self, img, face, cascadeIdentifier):
		self.face = face
		self.cascadeIdentifier = cascadeIdentifier
		self.image = HelperFunctions.cropImage(img, self)
		self.landmarks = []