import sys
import cv2
import HelperFunctions 

class Image:

	def __init__(self, img):
		self.img = img
		self.persons = None