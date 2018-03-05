import sys
import os
import cv2


def findEars(img):
	mac = True
	cascadePaths = []
	detected = []
	if mac:
		cascadePaths.append('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_mcs_rightear.xml')
		cascadePaths.append('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_mcs_leftear.xml')
	else:
		cascadePaths = 'haarcascade_mcs_leftear' #fix
	for cascadePath in cascadePaths:
		cascade = cv2.CascadeClassifier(cascadePath)
		detected += cascade.detectMultiScale(img, 1.1, 1).tolist()
	print(detected)
	for item in detected:
		(x,y,w,h) = item 
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	cv2.imshow('img', img)
	cv2.waitKey(0)


if __name__ == "__main__":
	findEars(cv2.imread('imgsInDatabase/img1.jpg'))