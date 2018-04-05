
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
