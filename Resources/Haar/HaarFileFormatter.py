import os
import cv2

def main():
	i = 0
	path = "/Users/admin/Documents/AudienceAttention/Pos/"
	for filename in os.listdir(path):
		if not filename.startswith("."):
			os.rename(path + filename, path + "img" + str(i) + ".jpg")
		i += 1

def main2():
	with open('samples.dat', 'w+') as f:
		path = "/Users/admin/Desktop/Haar_training/Positive_Images/"
		for filename in os.listdir(path):
			print(filename)
			if 'jpg' in filename:
				img = cv2.imread(path+filename)
				f.write("Positive_Images/" + filename + " 1 0 0 " + str(img.shape[0]) + " " + str(img.shape[1]) + "\r\n")



if __name__ == "__main__":
	main2()
