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
	with open('Haar/info.dat.txt', 'w+') as f:
		path = "/Users/admin/Documents/AudienceAttention/Haar/Pos/"
		for filename in os.listdir(path):
			print(filename)
			if filename.startswith("img") and filename != "img3.jpg":
				img = cv2.imread(path+filename)
				f.write("Pos/" + filename + " 1 0 0 " + str(img.shape[0]) + " " + str(img.shape[1]) + "\r\n")



if __name__ == "__main__":
	main2()
