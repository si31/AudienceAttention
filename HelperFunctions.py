import sys
import cv2
import numpy as np
import dlib

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