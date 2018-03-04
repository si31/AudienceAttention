import sys
import cv2
import numpy as np
from Person import Person
from Image import Image
import Main

listOfGoodComparisonsForHeadPose = [(4,1), (1,2), (2,3), (3,6), (6,9), (9,8), (8,7), (7,4)]
C=2.67
GAMMA=5.383

def compareLabels():
	img = Main.readFromDatabase(sys.argv[1])
	comparisons = []
	poseAreaAccuracy = 0
	for person in img.persons:
		if person.labels[0].humanFace:
			person.accumulateData()
			label = person.labels[0]
			thisComparison = ([person.poseArea, person.occlusion, person.postureArea, person.blur], [label.humanPoseAngle, label.humanOcclusion, label.humanPostureLR, label.humanMovement])
			comparisons.append(thisComparison)
			print(thisComparison)
			print(person.poseDistance)
	for comparison in comparisons:
		([a1,b1,c1,d1],[a2,b2,c2,d2]) = comparison
		if a1 == a2 or (a1,a2) in listOfGoodComparisonsForHeadPose or (a2,a1) in listOfGoodComparisonsForHeadPose:
			poseAreaAccuracy += 1
	print(poseAreaAccuracy / len(comparisons))


def SVMHeadPose():
	img = Main.readFromDatabase(sys.argv[1])
	trainData = []
	labels = []
	for person in img.persons:
		if person.labels[0].humanFace:
			trainData.append(person.poseParameters)
			labels.append(1 if person.labels[0].humanPoseAngle == 5 else -1)
	length = len(labels)
	svm = svmTrain(trainData[0:int(length*0.75)], labels[0:int(length*0.75)])
	print(svmTest(svm, trainData[int(length*0.75):length]))
	print(labels[int(length*0.75):length])


def svmTrain(trainData, labels):
	trainData = np.float32(trainData)#.reshape(-1,64)
	labels = np.array(labels)
	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_RBF)
	svm.setC(C)
	svm.setGamma(GAMMA)
	assert(len(trainData) == len(labels))
	svm.train(trainData, cv2.ml.ROW_SAMPLE, labels)
	svm.save('svm_data.dat')
	return svm


def svmTest(svm, testData):
	test = np.float32(testData)#.reshape(-1,64)
	result = svm.predict(test)
	return result

def main():
	#SVMHeadPose()
	compareLabels()

if __name__ == "__main__":
	main()
