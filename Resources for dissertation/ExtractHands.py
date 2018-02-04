import os
import cv2
import scipy.io as spio


def main():
	i = 0
	pathImages = "/Users/admin/Desktop/hand_dataset/training_dataset/training_data/images/"
	pathAnnotations = "/Users/admin/Desktop/hand_dataset/training_dataset/training_data/annotations/"
	for filename in os.listdir(pathImages):
		if not filename.startswith("."):
			filenameStart = filename.split('.')[0]
			img = cv2.imread(pathImages + filename)
			mat = spio.loadmat(pathAnnotations + filenameStart + '.mat', squeeze_me=True)
			print(mat)
			a = mat['M'] # array
		i += 1




if __name__ == "__main__":
	main()