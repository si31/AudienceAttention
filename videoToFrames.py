import os
import cv2
import ComputerVision

def saveImage(img, i ):
	print('Saving img to desktop...')
	cv2.imwrite("/Users/admin/desktop/" + str(i) + ".jpg", img)

def main():
	video = cv2.VideoCapture('split.mov')
	frameInterval = 10
	i = 0
	blurs = []
	while(video.isOpened()):
		ret, frame = video.read()
		if ret and frame is not None:
			i += 1
			if i % frameInterval != 0:
				continue
			#saveImage(frame, i)
			print('Frame ' + str(i//frameInterval) + ' completed.')
			blur = ComputerVision.blur(frame)
			blurs.append((i//10, blur))
		else:
			break
	video.release()
	print(blurs)
	result = sorted(blurs, key=lambda blur: blur[1]) 
	print(result)
	print([a for (a,b) in result])

if __name__ == "__main__":
	main()
