import os
import sys
import cv2
import numpy as np
from Person import Person, accumulateData, printData
from Image import Image
import HelperFunctions

listOfGoodComparisonsForHeadPose = [(4,1), (1,2), (2,3), (3,6), (6,9), (9,8), (8,7), (7,4)]

def compareLabels():
	img = HelperFunctions.readFromDatabase(sys.argv[1])
	comparisons = []
	poseAreaAccuracy = 0
	baseAccuracy = 0
	for person in img.persons:
		if person.labels[0].humanFace:
			accumulateData(person)
			label = person.labels[0]
			thisComparison = ([person.lookingForward, person.occlusion, person.postureArea, person.blur, person.earDetected], [label.humanPoseAngle, label.humanOcclusion, label.humanPostureLR, label.humanMovement])
			comparisons.append(thisComparison)
	for comparison in comparisons:
		([a1,b1,c1,d1,e1],[a2,b2,c2,d2]) = comparison
		if (a1 == 1 and a2 == 5) or (a1 == 0 and a2 != 5):
			poseAreaAccuracy += 1
		if a2 == 5:
			baseAccuracy += 1
	print(poseAreaAccuracy / len(comparisons))
	print(baseAccuracy / len(comparisons))

def collateData():
	data = []
	for file in os.listdir("Database/"):
		if 'jpg.txt' in file:
			img = HelperFunctions.readFromDatabase(file[:-4])
			for person in img.persons:
				accumulateData(person)
				if person.labels != []:
					label = person.labels[0]
					labelData = [label.humanFace, label.humanMovement, label.humanPoseAngle, label.humanPostureLR, label.humanOcclusion, label.humanEyeAngle]
					data.append((person.data, labelData))
	return data


def analyseData(data):
	# currently doing face detection accuracy
	totalDetections = len(data)
	count1 = 0
	count2 = 0
	count3 = 0
	count4 = 0
	for ([computerBlur, computerLookingForward, computerPostureLR, computerOcclusion], [humanFace, humanMovement, humanPoseAngle, humanPostureLR, humanOcclusion, humanEyeAngle]) in data:
		if humanFace != 0:
			if computerOcclusion != -1:
				if humanOcclusion == 1:
					if computerOcclusion == 1:
						count1 += 1
					else:
						count2 += 1
				else:
					if computerOcclusion != 1:
						count3 += 1
					else:
						count4 += 1

	print('Count1: ' + str(count1))
	print('Count2: ' + str(count2))
	print('Count3: ' + str(count3))
	print('Count4: ' + str(count4))
	print('Total: ' + str(totalDetections))


def main():
	analyseData(collateData())

if __name__ == "__main__":
	main()
