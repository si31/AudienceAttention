import cv2
import sys
import FaceDetection
import Image
import Person
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
	j=0
	for filename in os.listdir(os.getcwd()+'/source_videos'):
		if '.wmv' in filename or '.avi' in filename:
			cap = cv2.VideoCapture(os.getcwd()+'/source_videos/' + filename)
			i = 0
			interval = 20
			breakThru = False
			while(cap.isOpened()):
				ret, frame = cap.read()
				if i % interval == 0:
					#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
					if frame is None:
						break
					#detected = cascade.detectMultiScale(frame, 1.2, 1)
					#print(detected)
					#if len(detected) > 0:
					cv2.imshow('frame', frame)
						#hogDrawing = HOG.getHOG(cropImage(frame, detected[0]))
						#cv2.imshow('hog', hogDrawing)
					char = cv2.waitKey(0)
					if char == ord('s'):
						print('Saving...')
						cv2.imwrite("occluded_positive/frames/" + str(j) + "-" + str(int(i/interval)) + ".png", frame)
					elif char == ord('q'):
						breakThru = True
						break
				i += 1
		if breakThru:
			break
		j += 1

	cap.release()
	
if __name__ == "__main__":
	main()
