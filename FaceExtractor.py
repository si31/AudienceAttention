import cv2
import sys
import FaceDetection
import HOG
import os

def cropImage(img, face):
	(x,y,w,h) = face
	extra = 5
	x = max(0, x-extra)
	y = max(0, y-extra)
	h = h+extra*2
	w = w+extra*2
	return img[y:y+h, x:x+w]

def main():
	cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

	i=27
	for filename in os.listdir(os.getcwd()+'/occluded_positive/frames'):
		print(filename)
		if 'png' in filename:
			img = cv2.imread('occluded_positive/frames/' + filename)
			detected = cascade.detectMultiScale(img, 1.2, 1)
			print(detected)
			if len(detected) > 0:
				cropped = cropImage(img,detected[0])
				cv2.imshow('img', cropped)
				#hogDrawing = HOG.getHOG(cropped)
				#cv2.imshow('hog', hogDrawing)
				char = cv2.waitKey(0)
				if char == ord('q'):
					break
				elif char == ord('s'):
					print('Saving...')
					cv2.imwrite("occluded_positive/faces/" + str(i)+'.png', cropped)
					i += 1



	
if __name__ == "__main__":
	main()
