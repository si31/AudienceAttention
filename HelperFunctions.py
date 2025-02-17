import sys
import os
import cv2
import numpy as np
import dlib
import pickle
import math
from PIL import ImageDraw, ImageFont


def saveImage(img):
	print('Saving img to desktop...')
	cv2.imwrite("/Users/admin/desktop/saved.jpg", img)


def saveObject(obj):
	print('Saving obj to desktop...')
	with open("/Users/admin/desktop/saved.txt", 'wb') as f:
		pickle.dump(obj, f)


def saveToDatabase(img, imgName):
	print('Saving to database...')
	if img is None:
		print('Trying to save None object. Save failed.')
		return
	with open('Database/' + imgName + '.txt', 'wb') as f:
		pickle.dump(img, f)


def readFromDatabase(imgName):
	print('Reading from database...')
	with open('Database/' + imgName + '.txt', 'rb') as f:
		img = pickle.load(f)
	return img


def inDatabase(imgName):
	for file in os.listdir("Database/"):
		if file == imgName + '.txt':
			return True
	return False


#based on tutorial / post
def bbOverLapRatio(bb1, bb2):
	[bb1x1,bb1y1,w,h] = bb1
	[bb2x1,bb2y1,w,h] = bb2

	bb1x2 = bb1x1+w
	bb1y2 = bb1y1+h
	bb2x2 = bb2x1+w
	bb2y2 = bb2y1+h   

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1x1, bb2x1)
	y_top = max(bb1y1, bb2y1)
	x_right = min(bb1x2, bb2x2)
	y_bottom = min(bb1y2, bb2y2)

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1x2 - bb1x1) * (bb1y2 - bb1y1)
	bb2_area = (bb2x2 - bb2x1) * (bb2y2 - bb2y1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	return iou


def cropImage(img, person, extra):
	(x,y,w,h) = person.face
	x = max(0, x-extra)
	y = max(0, y-extra)
	h = h+extra*2
	w = w+extra*2
	return img[y:y+h, x:x+w]


def cropImageRatio(img, person, ratio):
	(x,y,w,h) = person.face
	extraX = int(ratio*(w//2))
	extraY = int(ratio*(h//2))
	x = max(0, x-extraX)
	y = max(0, y-extraY)
	wNew = w+extraX*2
	hNew = h+extraY*2
	if wNew + x < img.shape[1]:
		w = wNew
	else:
		w = img.shape[1] - x
	if hNew + y < img.shape[0]:
		h = hNew
	else:
		h = img.shape[0] - y
	return img[y:y+h, x:x+w]


def dlibRectToBB(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


def dlibBBToRect(x,y,w,h):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	return dlib.rectangle(x,y,x+w,y+h)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords


def calcDistance(point1, point2):
	(x1,y1) = point1
	(x2,y2) = point2
	return math.sqrt(math.pow(x2-x2,2)+math.pow(y2-y1,2))


FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotateImage(image, bb, val1, val2, ratio):
	(x,y,w,h) = bb
	(x,y,w,h) = (int(x*ratio) for x in list(bb))
	boxX = x + w + 10
	boxY = y
	boxW = 20
	boxH = 20
	cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 3)
	cv2.rectangle(image, (boxX, boxY), (boxX+boxW*2, boxY+boxH), (255,255,255), -1)
	cv2.putText(image, str(val1), (boxX, boxY+boxH-5), FONT, 0.5, (0,0,0), 1, cv2.LINE_AA)
	cv2.rectangle(image, (boxX, boxY+boxH+10), (boxX+boxW, boxY+2*boxH+10), (255,255,255), -1)
	cv2.putText(image, str(val2), (boxX, boxY+2*boxH+5), FONT, 0.5, (0,0,0), 1, cv2.LINE_AA)


def drawTextPIL(img, imagePIL):
	for person in img.persons:	
		(x,y,w,h) = person.face
		boxX = x + w + 10
		boxY = y
		boxW = w//4
		boxH = w//4
		draw = ImageDraw.Draw(imagePIL)
		draw.rectangle(((boxX, boxY), (boxX+boxW, boxY+boxH)), fill="gray")
		draw.text((boxY, boxY), 'hi', font=ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSans.ttf'))