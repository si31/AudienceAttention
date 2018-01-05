import sys
import cv2
import HelperFunctions 

class Image:

	def __init__(self, img):
		self.image = img
		self.persons = None
		self.blur = None