import sys
import os
import cv2


def findEars(img):
	mac = True
	cascadePath = ''
	if mac:
		cascadePath = '/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_mcs_leftear.xml'
	else:
		cascadePath = 'haarcascade_mcs_leftear' #fix
	cascade = cv2.CascadeClassifier(cascadePath)
	detected = cascade.detectMultiScale(img, 1.1, 1)
	print(detected)

if __name__ == "__main__":
	findEars(cv2.imread('imgsInDatabase/test1.jpg'))