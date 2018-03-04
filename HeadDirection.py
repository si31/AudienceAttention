import sys
import cv2
import numpy as np
import HelperFunctions
from Person import Person
import ComputerVision
import dlib
import math


#standard relative points on a 3D head model
#from learn opencv site
MODEL_POINTS_3D = np.array([(0.0, 0.0, 0.0),             # Nose tip
							(0.0, -300.0, -65.0),        # Chin #old 330
							(-250.0, 140.0, -135.0),     # Left eye left corner #old 140
							(250.0, 140.0, -135.0)      # Right eye right corner #old
							#(-150.0, -150.0, -125.0),    # Left Mouth corner
							#(150.0, -150.0, -125.0)      # Right mouth corner
						 	])

#my calcs from wiki page
"""
MODEL_POINTS_3D = np.array([(0.0, 0.0, 0.0),            # Sellion
							(0.0, -42.0, 22.0),        	# Nosetip
							(0.0, -117.5, 0.0),        	# Chin
							(-59.5, -5.0, -20.0),     	# Left eye left corner
							(59.5, -5.0, -20.0),      	# Right eye right corner
							#(-69.5, -21.0, -96.5),    	# Left face
							#(69.5, -21.0, -96.5),      	# Right face
							(0.0, -72.0, 20.0) 		  	# Middle of mouth
						 	])

#points from library attention
MODEL_POINTS_3D = np.array([(0.0, 0.0, 0.0),            # Sellion
							#(21.0, 0.0, -48.0),        	# Nosetip
							(0.0, 0.0, -133.0),        	# Chin
							(-20.0, -65.5, -5.0),     	# Left eye left corner
							(-20.0, 65.5, -5.0)      	# Right eye right corner
							#(-69.5, -21.0, -96.5),    	# Left face
							#(69.5, -21.0, -96.5),      	# Right face
							#(10.0, 0.0, -75.0) 		  	# Middle of mouth
							])
"""
							
#followed tutorial
def getPose(person, imgShape, mark=False):
	image = person.image
	size = imgShape
	 #from learn opencv site
	faceMarkers = np.array([tuple(person.landmarks[30]), 
							tuple(person.landmarks[8]), 
							tuple(person.landmarks[36]),
							tuple(person.landmarks[45])], 
							#tuple(person.landmarks[48]), 
							#tuple(person.landmarks[54])], 
							dtype="double")
	"""
	# from wiki calcs
	faceMarkers = np.array([tuple(person.landmarks[27]), 
							#tuple(person.landmarks[30]), 
							tuple(person.landmarks[8]),
							tuple(person.landmarks[36]), 
							tuple(person.landmarks[45])
							#tuple(person.landmarks[1]), 
							#tuple(person.landmarks[15])
							], dtype="double").reshape(4,1,2)
	print(faceMarkers.shape)
	"""
	focal_length = size[1]
	center = (size[1]//2, size[0]//2)
	camera_matrix = np.array([[focal_length, 0, center[0]],
								[0, focal_length, center[1]],
								[0,0,1]],
								dtype = "double")

	axis = np.float32([[0.0, 0.0, -1000.0], [1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]])
	dist_coeffs = np.zeros((4,1)) #assumes no lens distortion
	(success, rotation_vector, translation_vector) = cv2.solvePnP(MODEL_POINTS_3D, faceMarkers, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
	(nose_end_point2D, jacobian) = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
	person.poseParameters = [item for sublist in nose_end_point2D.tolist() for subsublist in sublist for item in subsublist]
	p1 = (int(faceMarkers[0][0]), int(faceMarkers[0][1]))
	p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
	p2relative = (p2[0]-p1[0], p2[1]-p1[1])
	person.poseAngle = math.atan2(p2relative[1], p2relative[0])
	person.poseDistance = math.sqrt(math.pow(p2relative[0], 2) + math.pow(p2relative[1], 2))

	if mark:
		#draw(image, tuple(faceMarkers[0][0]), nose_end_point2D)
		
		cv2.line(person.image, p1, p2, (255,0,0),2)
		
		for p in faceMarkers:
			cv2.circle(person.image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

def draw(img, origin, points):
	origin = tuple([int(p) for p in origin])[0:2]
	print(points)
	img = cv2.line(img, origin, tuple([int(p) for p in points[0][0]]), (255,0,0), 4)
	img = cv2.line(img, origin, tuple([int(p) for p in points[1][0]]), (0,255,0), 4)
	img = cv2.line(img, origin, tuple([int(p) for p in points[2][0]]), (0,0,255), 4)

def setup():
	testImg = cv2.imread('HeadPoseTest/test1.jpg')
	face = (0,0,testImg.shape[0],testImg.shape[1])
	person = Person(testImg, face, None)
	ComputerVision.faceLandmarks(person, mark=True)
	getPose(person, person.image.shape)
