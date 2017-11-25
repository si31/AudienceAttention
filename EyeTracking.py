import sys
import cv2
import numpy as np
import HelperFunctions

def findEyes(img, person):
	cascadePaths = ['/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_eye.xml']
	cascades = []
	for cascadePath in cascadePaths:
		cascades.append(cv2.CascadeClassifier(cascadePath))

	detectedAll = []
	totalSize = 0 
	for index, cascade in enumerate(cascades):
		detected = cascade.detectMultiScale(HelperFunctions.cropImage(img, person), 1.2, 1)
		print(detected)
		detectedAll = detected

	for item in detectedAll:
			((x,y,w,h),cascadeIdentifier) = item #could add a better way to choose the best box - biggest or average them
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
