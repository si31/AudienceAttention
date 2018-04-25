import os
import sys
import cv2
import numpy as np
import math
from sklearn import metrics, model_selection, svm as skSVM
import random
from Person import Person, accumulateData, printData
from Image import Image
import HelperFunctions
import GraphCreator
import krippendorff
import statistics
from sklearn.preprocessing import Imputer
import pickle


def collateData():
	data = []
	for file in os.listdir("Database/"):
		if 'jpg.txt' in file:
			img = HelperFunctions.readFromDatabase(file[:-4])
			print(len(img.persons))
			for person in img.persons:
				accumulateData(person)
				if person.labels != []:
					attentions = []
					if person.labels != []:
						label = person.labels[0]
						if label.humanFace == 1:
							for label in person.labels[1:]:
								attentions.append(label.humanAttention)
					if not badData(attentions):
						firstLabel = person.labels[0]
						if firstLabel.humanFace == 0:
							labelData = [0, 0, 0, 0, 0, 0, 0, 1]
						else:
							attention = 0
							i = 0
							for label in person.labels[1:]:
								attention += label.humanAttention
								i += 1
							divValue = len(person.labels[1:]) if len(person.labels[1:]) >= 1 else 1
							attention = attention / divValue
							labelData = [1, firstLabel.humanMovement, firstLabel.humanPoseAngle, firstLabel.humanPostureLR, 
										firstLabel.humanOcclusion, firstLabel.humanEyeAngle, attention, i]

						data.append((person.data, labelData, person.face, person.leftLooking, person.attention))
	return data


def collateDataKripp():
	labelSet = []
	for file in os.listdir("Database/"):
		if 'jpg.txt' in file:
			img = HelperFunctions.readFromDatabase(file[:-4])
			for person in img.persons:
				accumulateData(person)
				attentions = []
				if person.labels != []:
					label = person.labels[0]
					if label.humanFace == 1:
						for label in person.labels[1:]:
							attentions.append(label.humanAttention)
				labelSet.append(attentions)
	return labelSet

def collateDataMLFullArray(): #inc openpose
	labelSet = []
	data = []
	for file in os.listdir("Database/"):
		if 'jpg.txt' in file:
			img = HelperFunctions.readFromDatabase(file[:-4])
			for person in img.persons:
				accumulateData(person)
				attentions = []
				if person.labels != []:
					label = person.labels[0]
					if label.humanFace == 1:
						for label in person.labels[1:]:
							attentions.append(label.humanAttention)
				if not badData(attentions):# and person.data[1:2] != [-1]:
					labelSet.append(attentions)
					data.append(person.data)
					"""
					if person.leftLooking is not None:
					labelSet.append(attentions)
					dataPoint = [item for sublist in person.leftLooking for item in sublist]
					new = []
					for data in dataPoint:
						if isinstance(data, list):
							print(data)
							new += [item for item in data]
						else:
							new.append(data)
					print(new)
					data.append(new)#erson.data
					"""

	return data, labelSet


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
						attention = 0.0
						for label in person.labels[1:]:
							attention += label.humanAttention
						divValue = len(person.labels[1:]) if len(person.labels[1:]) >= 1 else 1
						attention = attention / divValue
						data.append(person.data[1:4])
						labels.append(attention)
	return data, labels


def analyseData(data):
	# currently doing face detection accuracy
	totalDetections = len(data)
	postureCorrect = 0
	postureIncorrect = 0
	postureOcclusionNA = 0
	falseDetections = 0
	occlusionCorrect = 0
	occlusionIncorrect = 0
	poseCorrect = 0
	poseIncorrect = 0
	faces = 0
	var1 = 0
	var2 = 0
	var3 = 0
	var4 = 0

	qualityComparison = []

	i=0
	poseValues = [[],[],[]]
	graphValues = [[],[]]
	for ([computerLookingForward, computerPostureArea, computerOcclusion, poseAngle, poseDistance, postureLR], [humanFace, humanMovement, humanPoseAngle, humanPostureLR, humanOcclusion, humanEyeAngle, humanAttention, noLabels], face, points, predAttention) in data:
		if humanFace == 1:
			
			(x,y,w,h) = face
			"""
			if predAttention > 0:
				if computerLookingForward == 1:
					var1 += 1
				else:
					var2 += 1
			else:
				if computerLookingForward == 1:
					var3 += 1
				else:
					var4 += 1
			"""

			if computerPostureArea != -1:
				
				if predAttention <= 0:
					if computerPostureArea == 0:
						var1 += 1
					elif computerPostureArea == 1:
						var2 += 1
					else:
						var3 += 1
				"""
				if predAttention <= 0:
					if computerOcclusion == 1:
						var1 += 1
					else:
						var2 += 1
				"""

				graphValues[0].append(postureLR)
				graphValues[1].append(predAttention)

				if computerPostureArea == humanPostureLR:
					postureCorrect += 1
				else:
					postureIncorrect += 1
				if computerOcclusion == humanOcclusion:
					qualityComparison.append((w*h, True))
					occlusionCorrect += 1
				else:
					qualityComparison.append((w*h, False))
					occlusionIncorrect += 1
			else:
				postureOcclusionNA += 1
			if computerLookingForward == 1 and humanPoseAngle == 5 or computerLookingForward == 0 and humanPoseAngle != 5:
				poseCorrect += 1
			else:
				poseIncorrect += 1
			faces += w*h
			i += 1
		else:
			falseDetections += 1
	print('postureCorrect: ' + str(postureCorrect))
	print('postureIncorrect: ' + str(postureIncorrect))
	print('occlusionCorrect: ' + str(occlusionCorrect))
	print('occlusionIncorrect: ' + str(occlusionIncorrect))
	print('NA: ' + str(postureOcclusionNA))
	print('poseCorrect: ' + str(poseCorrect))
	print('poseIncorrect: ' + str(poseIncorrect))
	print('False Negatives: ' + str(falseDetections))
	print('Total: ' + str(totalDetections))

	print(":")
	print((var1,var2,var3,var4))
	cat1 = []
	cat2 = []
	cat3 = []
	cat4 = []
	cat5 = []
	for (size, val) in qualityComparison:
		if size < 1000:
			cat1.append(val)
		elif size < 3000:
			cat2.append(val)
		elif size < 5000:
			cat3.append(val)
		elif size < 7000:
			cat4.append(val)
		else:
			cat5.append(val)
	print((len(cat1), len(cat2), len(cat3), len(cat4), len(cat5)))
	print(sum(cat1), sum(cat2), sum(cat3), sum(cat4), sum(cat5))
	#GraphCreator.createGraph(graphValues[1], graphValues[0], '', '', '')
	return [postureCorrect, postureIncorrect, occlusionCorrect, occlusionIncorrect, postureOcclusionNA, poseCorrect, poseIncorrect, totalDetections-falseDetections]


def svmTrainSK(trainData, labels, classification=True, save=False):
	trainData = np.float32(trainData)
	labels = np.float32(labels)
	classifier = None
	print(trainData)
	if classification:
		param_grid = [{'C': [1, 10, 100, 1000, 10000, 100000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']}]	
		classifier = skSVM.SVC()
		classifier = model_selection.GridSearchCV(classifier, param_grid)
	else:
		classifier = skSVM.SVR()
	classifier.fit(trainData, labels)
	if classification:
		print(classifier.best_params_)
	if save:
		with open('model.pkl', 'wb') as f:
			pickle.dump(classifier, f) 
	return classifier


def svmTestSK(classifier, testData):
	testData = np.float32(testData)
	result = classifier.predict(testData)
	return result


def MLClassification(data, labels, zipped):
	data, labels = zip(*zipped)
	labels = [1 if label >= 0 else -1 for label in labels]
	trainData = data[0:(len(data)*3)//4]
	testData = data[(len(data)*3)//4:]
	trainLabels = labels[0:(len(data)*3)//4]
	testLabels = labels[(len(data)*3)//4:]
	classifier = svmTrainSK(trainData, trainLabels)
	result = svmTestSK(classifier, testData)
	result = result.tolist()
	theSame = 0
	for i in range(len(testLabels)):
		if testLabels[i] == int(result[i]):
			theSame += 1

	print(theSame)
	print(len(testLabels))
	print(str(theSame/len(testLabels)))
	print(' ')
	MLClassificationEval(data, labels)
	print(metrics.confusion_matrix(testLabels, result))


def MLRegression(data, labels, zipped):
	data, labels = zip(*zipped)
	trainData = data[0:(len(data)*3)//4]
	testData = data[(len(data)*3)//4:]
	trainLabels = labels[0:(len(data)*3)//4]
	testLabels = labels[(len(data)*3)//4:]
	classifier = svmTrainSK(trainData, trainLabels, classification=False)
	result = svmTestSK(classifier, testData)
	result = result.tolist()
	theSame = 0
	meanDiff = 0.0
	#change for comparison
	testLabels = [1 if label >= 0 else 0 for label in testLabels]
	result = [1 if label >= 0 else 0 for label in result]
	for i in range(len(testLabels)):
		meanDiff += math.pow(testLabels[i]-result[i],2)
		if testLabels[i] == result[i]:
			theSame += 1
	print(theSame)
	print(len(testLabels))
	meanDiff = meanDiff / len(testLabels)
	MLRegressionEval(data, labels)


def MLClassificationEval(data, labels):
	kfold = model_selection.KFold(n_splits=5, random_state=7)
	model = skSVM.SVC(gamma=0.001, C=100.)
	scoring = 'roc_auc' #roc_auc or r2
	results = model_selection.cross_val_score(model, data, labels, cv=kfold, scoring=scoring)
	print(results.mean())


def MLRegressionEval(data, labels):
	kfold = model_selection.KFold(n_splits=5, random_state=7)
	scoring = 'neg_mean_absolute_error' #neg_mean_absolute_error or r2
	model = skSVM.SVR()
	score = model_selection.cross_val_score(model, data, labels, cv=kfold, scoring=scoring)
	print(score)
	print(score.mean())


def performKFold(k, data, labels, classification):
	alltestlabels = []
	allresultlabels = []
	if classification:
		labels = [1 if label >= 0 else -1 for label in labels]
	for i in range(k):
		l = len(data)
		w = len(data) / k
		trainData = data[:int(i*w)] + data[int((i+1)*w):]
		testData = data[int(i*w):int((i+1)*w)]
		trainLabels = labels[:int(i*w)] + labels[int((i+1)*w):]
		testLabels = labels[int(i*w):int((i+1)*w)]
		classifier = svmTrainSK(trainData, trainLabels, classification=classification)
		result = svmTestSK(classifier, testData)
		alltestlabels += testLabels
		allresultlabels += result.tolist()
	theSame = 0
	if not classification:
		allresultlabels = [1 if label >= 0 else 0 for label in allresultlabels]
		alltestlabels = [1 if label >= 0 else 0 for label in alltestlabels]	
	for i in range(len(alltestlabels)):
		if alltestlabels[i] == allresultlabels[i]:
			theSame += 1
	print('hi')
	print(theSame)
	print(len(allresultlabels))
	print(metrics.confusion_matrix(alltestlabels, allresultlabels))
	print(metrics.f1_score(alltestlabels, allresultlabels))


def badData(label):
	valueCount = [label.count(2) + label.count(1), label.count(0), label.count(-1) + label.count(-2)]
	return not ((4 in valueCount or 5 in valueCount) and not valueCount == [0,0,0])


def performKripp():
	labelSet = collateDataKripp()
	valueSet = []
	badOnes = 0
	for label in labelSet:
		newValue = [label.count(2) + label.count(1), label.count(0), label.count(-1) + label.count(-2)]
		if not badData(label):
			valueSet.append(newValue)
		if badData(label):
			badOnes += 1
	print(valueSet)
	valueSet = np.array(valueSet)
	krippVal = krippendorff.alpha(value_counts=valueSet, level_of_measurement='ordinal')
	print(krippVal)
	print(badOnes)


def main():
	#performKripp()
	return analyseData(collateData())
	
	data, labels = collateDataMLFullArray()
	labels = [statistics.mean(label) for label in labels]
	imp = Imputer(missing_values=0.0, strategy='mean', axis=0)
	imp.fit(data)
	data = imp.transform(data)
	zipped = list(zip(data, labels))
	random.shuffle(zipped)
	#MLClassification(data, labels, zipped)
	#MLRegression(data, labels, zipped)
	data, labels = zip(*zipped)
	performKFold(10, data, labels, True)
	performKFold(10, data, labels, False)

	#for training actual model used
	#svmTrainSK(data, labels, classification=False, save=True)
	

if __name__ == "__main__":
	main()

