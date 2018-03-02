import sys
import cv2
import numpy as np
import uuid
import base64
import math

from skimage.feature import hog
from skimage import data, exposure
from random import shuffle

SZ = 20
C=2.67
GAMMA=5.383

height = 32
width = 32

cellsize = 4

viewRatio = 3

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def convertImage(img):
	img = cv2.resize(img, (width,height), interpolation=cv2.INTER_LINEAR);
	img = np.float32(img) / 255.0

	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	#calculate x and y gradient 
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

	#save
	if False:
		cv2.imwrite("Gradients/" + str(i) + "x.png", gx * 255.0);
		cv2.imwrite("Gradients/" + str(i) + "y.png", gy * 255.0);

	#convert to vectors
	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)


	#reduce to 5x5 cells (calculate histograms)
	histogramBins = [0] * 9 #0-39, 40-79, ..., 320-359
	hog = []
	for j in range(0,width//cellsize):#switched
		for i in range(0,height//cellsize):
			histogramBins = [0]*9 #0-39, 40-79, ..., 320-359
			for x in range(0,cellsize):
				for y in range(0,cellsize):
					histogramBins[int(angle[i*cellsize+x, j*cellsize+y] / 40)] += mag[i*cellsize+x, j*cellsize+y]
			hog += (histogramBins)

	#normalisation
	for i in range(0, width//(cellsize*2) + height//(cellsize*2)):
		normalisationRange = hog[i*18:(i+1)*18] + hog[(i*18+(9*(width//cellsize))):i*18+(9*(width//cellsize))]
		print(normalisationRange)
		length = 0
		for val in normalisationRange:
			length += math.pow(val,2)
		length = math.sqrt(length)
		if length == 0:
			length = 1
		for j in range(i*18, (i+1)*18):
			hog[j] = hog[j] / length
		normalisationRange = hog[i*18:(i+1)*18]
		print(normalisationRange)


	return np.array(hog)

"""
def openCVHOG(img):
		#resize
		img = cv2.resize(img, (width,height), interpolation=cv2.INTER_LINEAR);

		#img = np.float32(img) / 255.0
		#img = np.uint8(img)

		#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		hog = cv2.HOGDescriptor((width,height), (cellsize,cellsize), (cellsize,cellsize), (cellsize,cellsize), 9)
		#svm = pickle.load(open("svm.pickle"))
		#hog.setSVMDetector( np.array(svm) )
		#del svm
		#found, w = hog.detectMultiScale(img)
		output = hog.compute(img)
		return output
"""


def svmTrain(trainData, labels):
	#deskewed = [map(deskew,row) for row in trainData]
	for index, img in enumerate(trainData):
		if img is None:
			print('crap')
			print(index)
	hogdata = [convertImage(img).flatten() for img in trainData]
	hogdata = np.float32(hogdata)#.reshape(-1,64)
	labels = np.array(labels)
	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	# Set SVM Kernel to Radial Basis Function (RBF) 
	svm.setKernel(cv2.ml.SVM_RBF)
	# Set parameter C
	svm.setC(C)
	# Set parameter Gamma
	svm.setGamma(GAMMA)
	#trainingSequence = [(hogdata[i],labels[i]) for i in range(0, len(labels))]
	assert(len(hogdata) == len(labels))
	print(hogdata.shape)
	#hogdata = hogdata.reshape(-1,64)
	print(hogdata.shape)
	svm.train(hogdata, cv2.ml.ROW_SAMPLE, labels)
	svm.save('svm_data.dat')
	return svm


def svmTest(svm, testData):
	#deskewed = [map(deskew,row) for row in test_cells]
	hogdata = [convertImage(img).flatten() for img in testData]
	hogdata = np.float32(hogdata)#.reshape(-1,64)
	result = svm.predict(hogdata)
	return result


def collectImages():
	imagesPosTrain = []
	imagesPosTest = []
	imagesNegTrain = []
	imagesNegTest = []
	num_pos = 149
	num_neg = 196
	for i in range(0, math.floor((num_pos/10)*9)):
		#load image
		img = cv2.imread('occluded_positive/faces/' + str(i) + '.png')
		#images 
		if img is None:
			continue
		imagesPosTrain.append(img)
	for i in range(math.floor((num_pos/10)*9), num_pos+1):
		#load image
		img = cv2.imread('occluded_positive/faces/' + str(i) + '.png')
		#images 
		if img is None:
			continue
		imagesPosTest.append(img)
	for i in range(0, math.floor((num_neg/10)*9)):
		#load image
		img = cv2.imread('occluded_negative/faces/' + str(i) + '.png')
		if img is None:
			continue
		#images 
		imagesNegTrain.append(img)
	for i in range(math.floor((num_neg/10)*9), num_neg+1):
		#load image
		img = cv2.imread('occluded_negative/faces/' + str(i) + '.png')	
		if img is None:
			continue
		#images 
		imagesNegTest.append(img)
	return imagesPosTrain, imagesPosTest, imagesNegTrain, imagesNegTest


def draw_hog(img, hog):
	hog = hog.flatten()
	inflatedCellsize = cellsize*viewRatio
	h, w = img.shape[0], img.shape[1]
	halfcell = inflatedCellsize/2
	horiCells, vertCells = w//inflatedCellsize, h//inflatedCellsize
	assert(horiCells == w/inflatedCellsize)
	maxv = np.max(hog)
	for x in range(0, horiCells):
		for y in range(0, vertCells):
			px, py = int(x*inflatedCellsize + inflatedCellsize/2), int(y*inflatedCellsize + inflatedCellsize/2)
			featNums = x*vertCells*9+y*9
			feat = hog[featNums:featNums+9]			 
			#maxv = np.max(feat)
			fdraw = []
			for i in range(0, 9):
				angle = i*40+20+90
				x1 = int(round((feat[i]/maxv)*halfcell*np.cos(np.deg2rad(angle))))
				y1 = int(round((feat[i]/maxv)*halfcell*np.sin(np.deg2rad(angle))))
				gv = int(round(255*feat[i]/maxv))
				fdraw.append((x1,y1,gv))
				 
			for x1,y1,gv in fdraw:
				cv2.line(img, (px-x1,py+y1), (px+x1,py-y1), (255, 255, 255), 1, 8)

def draw_mag(img,hog):
	hog = hog.flatten()
	inflatedCellsize = cellsize*viewRatio
	h, w = img.shape[0], img.shape[1]
	halfcell = inflatedCellsize/2
	horiCells, vertCells = w//inflatedCellsize, h//inflatedCellsize
	assert(horiCells == w/inflatedCellsize)
	maxv = np.max(hog)
	for x in range(0, horiCells):
		for y in range(0, vertCells):
			px, py = int(x*inflatedCellsize + inflatedCellsize/2), int(y*inflatedCellsize + inflatedCellsize/2)
			featNums = x*vertCells*9+y*9
			feat = hog[featNums:featNums+9]			 
			#maxv = np.max(feat)
			fdraw = []
			for i in range(0, 9):
				angle = i*40+20+90
				x1 = int(round((feat[i]/maxv)*halfcell*np.cos(np.deg2rad(angle))))
				y1 = int(round((feat[i]/maxv)*halfcell*np.sin(np.deg2rad(angle))))
				gv = int(round(255*feat[i]/maxv))
				fdraw.append((x1,y1,gv))
				 
			for x1,y1,gv in fdraw:
				cv2.circle(img, (px,py), int((gv/255)*inflatedCellsize), (gv, gv, gv), 1)

def getHOG(img):
	img = cv2.resize(img, (width,height), interpolation=cv2.INTER_LINEAR);
	hog = convertImage(img)
	hogDrawing = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#np.zeros(img.shape)
	hogDrawing = cv2.resize(hogDrawing, (width*viewRatio,height*viewRatio), interpolation=cv2.INTER_LINEAR);
	draw_hog(hogDrawing, hog)
	return hogDrawing

def disturbance(img, hog):
	hog = hog.flatten()
	print(hog)
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			if (x-1)> img.shape[0] and hog[x-1,y] < hog[x,y]:
				print('hi')
	
	
	pass

def main():
	print('Start...')
	
	img = cv2.imread('occluded_positive/faces/31.png')
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
	half = False
	global height
	imagesPosTrain, imagesPosTest, imagesNegTrain, imagesNegTest = collectImages()
	print(len(imagesPosTrain))
	if half:
		height = int(height/2)
		imagesPosTrain = shuffle([img[int(img.shape[1]/2):img.shape[1], 0:img.shape[0]] for img in imagesPosTrain])
		imagesPosTest = shuffle([img[int(img.shape[1]/2):img.shape[1], 0:img.shape[0]] for img in imagesPosTest])
		imagesNegTrain = shuffle([img[int(img.shape[1]/2):img.shape[1], 0:img.shape[0]] for img in imagesNegTrain])
		imagesNegTest = shuffle([img[int(img.shape[1]/2):img.shape[1], 0:img.shape[0]] for img in imagesNegTest])
	if sys.argv[1] == 'view':
		for img in imagesPosTrain:
			hog = getHOG(img)
			cv2.imshow('hog', hog)
			char = cv2.waitKey(0)
			if char == ord('q'):
				break
	else:
		labels = [1 for i in imagesPosTrain] + [-1 for i in imagesNegTrain]
		svm = svmTrain(imagesPosTrain + imagesNegTrain, labels)
		result = svmTest(svm, imagesPosTest)
		print(result[1].tolist())
	"""
	
	
if __name__ == "__main__":
	main()
