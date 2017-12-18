import sys
import cv2
import numpy as np
import HelperFunctions
from Person import Person
import ComputerVision
import dlib


#standard relative points on a 3D head model
MODEL_POINTS_3D = np.array([(0.0, 0.0, 0.0),             # Nose tip
							(0.0, -330.0, -65.0),        # Chin
							(-225.0, 170.0, -135.0),     # Left eye left corner
							(225.0, 170.0, -135.0),      # Right eye right corne
							(-150.0, -150.0, -125.0),    # Left Mouth corner
							(150.0, -150.0, -125.0)      # Right mouth corner
						 	])

#camera calibration


#followed tutorial
def getPose(person):
	image = person.image
	size = image.shape
	faceMarkers = np.array([tuple(person.landmarks[33]), tuple(person.landmarks[8]), tuple(person.landmarks[36]),
					tuple(person.landmarks[45]), tuple(person.landmarks[48]), tuple(person.landmarks[54])], dtype="double")
	focal_length = size[1] #needs to be replaced by the whole image's width
	center = (size[1]/2, size[0]/2)
	camera_matrix = np.array([[focal_length, 0, center[0]],
								[0, focal_length, center[1]],
								[0,0,1]],
								dtype = "double")

	dist_coeffs = np.zeros((4,1)) #assumes no lens distortion
	(success, rotation_vector, translation_vector) = cv2.solvePnP(MODEL_POINTS_3D, faceMarkers, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
	(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

	for p in faceMarkers:
		cv2.circle(person.image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

	p1 = (int(faceMarkers[0][0]), int(faceMarkers[0][1]))
	p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
	cv2.line(person.image, p1, p2, (255,0,0),2)
	cv2.imshow('name', person.image)
	cv2.waitKey(0)

def setup():
	testImg = cv2.imread('HeadPoseTest/test1.jpg')
	face = (0,0,testImg.shape[0],testImg.shape[1])
	person = Person(testImg, face, None)
	ComputerVision.faceLandmarks(person, mark=True)
	getPose(person)


if __name__ == "__main__":
	setup()