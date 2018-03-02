import sys
import cv2

import Image
import Person

import HOG
import HelperFunctions

def detectEyesOpen():
	img = cv2.imread('imgsInDatabase/'+sys.argv[1])

	cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_eye.xml')

	detected = cascade.detectMultiScale(img, 1.1, 1)

	hogView1 = None

	for face in detected:
		face = face.astype(int)
		face = face.tolist()
		(x,y,w,h) = face
		eye1 = img[y:y+h, x:x+w]
		hogView1 = HOG.getHOG(eye1)
		#cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)



	cv2.imshow('img', img)
	cv2.imshow('hogeye', hogView1)
	cv2.waitKey(0)




if __name__ == "__main__":
	detectEyesOpen()