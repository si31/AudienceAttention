import os
import sys
import cv2
import numpy as np
import math
from sklearn import svm as skSVM

from Person import Person, accumulateData, printData
from Image import Image
import HelperFunctions
import GraphCreator


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
					firstLabel = person.labels[0]
					if firstLabel.humanFace == 0:
						labelData = [0, 0, 0, 0, 0, 0, 0]
					else:
						attention = 0
						for label in person.labels[1:]:
							attention += label.humanAttention
						divValue = len(person.labels[1:]) if len(person.labels[1:]) >= 1 else 1
						attention = attention / divValue
						labelData = [1, firstLabel.humanMovement, firstLabel.humanPoseAngle, firstLabel.humanPostureLR, 
									firstLabel.humanOcclusion, firstLabel.humanEyeAngle, attention]

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
	postureCorrect = 0
	postureIncorrect = 0
	postureOcclusionNA = 0
	count4 = 0
	occlusionCorrect = 0
	occlusionIncorrect = 0
	poseCorrect = 0
	poseIncorrect = 0
	faces = 0
	i=0
	poseValues = [[],[],[]]
	blurValues = [[],[]]
	for ([computerBlur, computerLookingForward, computerPostureArea, computerOcclusion, poseAngle, poseDistance, blur], [humanFace, humanMovement, humanPoseAngle, humanPostureLR, humanOcclusion, humanEyeAngle, humanAttention], face) in data:
		if humanFace == 1:
			if computerPostureArea is not None:		
				if computerPostureArea == humanPostureLR:
					postureCorrect += 1
				else:
					postureIncorrect += 1
			else:
				postureOcclusionNA += 1
			if computerOcclusion != -1:
				if computerOcclusion == humanOcclusion:
					occlusionCorrect += 1
				else:
					occlusionIncorrect += 1
			if computerLookingForward == 1 and humanPoseAngle == 5:
				poseCorrect += 1
			else:
				poseIncorrect += 1
			if humanAttention == 1:
				count4 += 1
			(x,y,w,h) = face
			faces += w*h
			i+=1
			poseValues[0].append(i)
			poseValues[1].append(poseDistance)
			poseValues[2].append((humanPoseAngle == 5))
			blurValues[0].append((blur*10000)/(w*h))
			print(blur)
			blurValues[1].append(humanMovement)
			print('a')
		else:
			count4 += 1

	#print(math.sqrt(faces/(totalDetections-count4)))

	print('postureCorrect: ' + str(postureCorrect))
	print('postureIncorrect: ' + str(postureIncorrect))
	print('occlusionCorrect: ' + str(occlusionCorrect))
	print('postureIncorrect: ' + str(postureIncorrect))
	print('occlusionIncorrect: ' + str(occlusionIncorrect))
	print('poseCorrect: ' + str(poseCorrect))
	print('poseIncorrect: ' + str(poseIncorrect))
	print('False Negatives: ' + str(count4))
	print('Total: ' + str(totalDetections))
	GraphCreator.createGraph(poseValues[0], blurValues[0], '', '', '', labels=blurValues[1])

	return [postureCorrect, postureIncorrect, occlusionCorrect, occlusionIncorrect, postureOcclusionNA, poseCorrect, poseIncorrect]


SZ = 20
C=2.67
GAMMA=5.383


def svmTrainCV(trainData, labels):
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


def svmTestCV(svm, testData):
	testData = np.float32(testData)
	result = svm.predict(testData)
	return result


def svmTrainSK(trainData, labels):
	trainData = np.float32(trainData)
	labels = np.float32(labels)
	classifier = skSVM.SVC(gamma=0.001, C=100.)
	classifier.fit(trainData, labels)  
	return classifier


def svmTestSK(classifier, testData):
	testData = np.float32(testData)
	result = classifier.predict(testData)
	return result


def headAngle():
	collateData()


def main():
	analyseData(collateData())
	"""
	data, labels = collateDataML()
	labels = [1 if label == 1 else -1 for label in labels]
	trainData = data[0:(len(data)*7)//8]
	testData = data[(len(data)*7)//8:]
	trainLabels = labels[0:(len(data)*7)//8]
	testLabels = labels[(len(data)*7)//8:]	
	classifier = svmTrainSK(trainData, trainLabels)
	result = svmTestSK(classifier, testData)
	print(classifier.get_params())
	print(result)
	result = result.tolist()
	theSame = 0
	for i in range(len(testLabels)):
		print((testData[i], testLabels[i], result[i]))
		if testLabels[i] == int(result[i]):
			theSame += 1

	print(theSame)
	print(len(testLabels))
	print(str(theSame/len(testLabels)))
	"""

if __name__ == "__main__":
	main()
