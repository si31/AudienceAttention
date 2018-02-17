import sys
import os
import numpy as np
import cv2
import math

def collectImages():
	imagesPosTrain = []
	imagesPosTest = []
	imagesNegTrain = []
	imagesNegTest = []
	num_pos = 149
	num_neg = 196
	for i in range(0, math.floor((num_pos/10)*9)):
		img = cv2.imread('occluded_positive/faces/side/' + str(i) + '.png')
		if img is None:
			continue
		imagesPosTrain.append(img)
	for i in range(math.floor((num_pos/10)*9), num_pos+1):
		img = cv2.imread('occluded_positive/faces/side/' + str(i) + '.png')
		if img is None:
			continue
		imagesPosTest.append(img)
	for i in range(0, math.floor((num_neg/10)*9)):
		img = cv2.imread('occluded_negative/faces/' + str(i) + '.png')
		if img is None:
			continue 
		imagesNegTrain.append(img)
	for i in range(math.floor((num_neg/10)*9), num_neg+1):
		img = cv2.imread('occluded_negative/faces/' + str(i) + '.png')	
		if img is None:
			continue
		imagesNegTest.append(img)
	return imagesPosTrain, imagesPosTest, imagesNegTrain, imagesNegTest

def detectFromCascade(images):
	cascadePath = '/Users/admin/Documents/AudienceAttention/haar_training/haarcascade/cascade.xml'
	cascade = cv2.CascadeClassifier(cascadePath)
	results = []
	for img in images:
		detected = cascade.detectMultiScale(img, 1.01, 1)
		results.append(len(detected))

	return results

def main():
	print('Start...')
	"""
	img = cv2.imread('occluded_positive/faces/25.png')
	img = cv2.resize(img, (width,height), interpolation=cv2.INTER_LINEAR);
	hog = convertImage(img)
	blackBG = True
	if blackBG:
		img = np.zeros(img.shape)
	img = cv2.resize(img, (width*viewRatio,height*viewRatio), interpolation=cv2.INTER_LINEAR);
	draw_hog(img, hog)
	disturbance(img, hog)
	cv2.imshow('img', img)
	char = cv2.waitKey(0)
	if (char == 115):
		print('Saved...')
		cv2.imwrite('save.png', img)
	"""
	imagesPosTrain, imagesPosTest, imagesNegTrain, imagesNegTest = collectImages()
	
	if sys.argv[1] == 'view':
		pass
	else:
		results = detectFromCascade(imagesNegTrain)
		print(results)
	
	
	
if __name__ == "__main__":
	main()