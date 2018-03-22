import os
import sys
import cv2
import numpy as np
import math

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
					data.append((person.data, labelData, person.face))
	return data


def collateDataML():
	data = []
	labels = []
	for file in os.listdir("Database/"):
		if 'jpg.txt' in file:
			img = HelperFunctions.readFromDatabase(file[:-4])
			for person in img.persons:
				accumulateData(person)
				if person.labels != []:
					label = person.labels[0]
					if label.humanFace == 1:
						data.append(person.data[1:])
						labels.append(label.humanAttention)
	return data, labels


def analyseData(data):
	# currently doing face detection accuracy
	totalDetections = len(data)
	count1 = 0
	count2 = 0
	count3 = 0
	count4 = 0
	faces = 0
	for ([computerBlur, computerLookingForward, computerPostureArea, computerOcclusion], [humanFace, humanMovement, humanPoseAngle, humanPostureLR, humanOcclusion, humanEyeAngle], face) in data:
		if humanFace == 1:
			""" posture
			if computerPostureArea is not None:		
				if computerPostureArea == humanPostureLR:
					count1 += 1
				else:
					count2 += 1
			else:
				count3 += 1
			"""
			""" occlusion
			if computerOcclusion != -1:
				if computerOcclusion == humanOcclusion:
					count1 += 1
				else:
					count2 += 1
			else:
				count3 += 1
			"""
			if computerLookingForward == 1 and humanPoseAngle == 5:
				count1 += 1
			else:
				count2 += 1
			(x,y,w,h) = face
			faces += w*h
		else:
			count4 += 1

	print(math.sqrt(faces/(totalDetections-count4)))

	print('Count1: ' + str(count1))
	print('Count2: ' + str(count2))
	print('Count3: ' + str(count3))
	print('Count4: ' + str(count4))
	print('Total: ' + str(totalDetections))


SZ = 20
C=2.67
GAMMA=5.383


def svmTrain(trainData, labels):
	trainData = np.float32(trainData)
	labels = np.array(labels)
	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	# Set SVM Kernel to Radial Basis Function (RBF) 
	svm.setKernel(cv2.ml.SVM_RBF)
	# Set parameter C
	svm.setC(C)
	# Set parameter Gamma
	svm.setGamma(GAMMA)
	assert(len(trainData) == len(labels))
	svm.train(trainData, cv2.ml.ROW_SAMPLE, labels)
	svm.save('svm_data.dat')
	return svm


def svmTest(svm, testData):
	testData = np.float32(testData)
	result = svm.predict(testData)
	return result


def main():
	#analyseData(collateData())
	data, labels = collateDataML()
	labels = [1 if label == 1 else -1 for label in labels]
	trainData = data[0:(len(data)*7)//8]
	testData = data[(len(data)*7)//8:]
	trainLabels = labels[0:(len(data)*7)//8]
	testLabels = labels[(len(data)*7)//8:]	
	svm = svmTrain(trainData, trainLabels)
	result = svmTest(svm, testData)
	result = result[1].tolist()
	theSame = 0
	for i in range(len(testLabels)):
		print((testData[i], testLabels[i], result[i]))
		if testLabels[i] == int(result[i][0]):
			theSame += 1

	print(theSame)
	print(len(testLabels))
	print(str(theSame/len(testLabels)))


if __name__ == "__main__":
	main()
