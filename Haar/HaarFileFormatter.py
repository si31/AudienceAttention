import os

def main():
	i = 0
	path = "/Users/admin/Documents/AudienceAttention/Pos/"
	for filename in os.listdir(path):
		if not filename.startswith("."):
			os.rename(path + filename, path + "img" + str(i) + ".jpg")
		i += 1

def main2():
	with open('Pos/bg.txt', 'w+') as f:
		for i in range(0, 51):
			f.write("img/img" + str(i) + ".jpg\r\n")



if __name__ == "__main__":
	main()
