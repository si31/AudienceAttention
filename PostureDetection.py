import sys
sys.path.append('/home/simon/PyOpenPose/build/PyOpenPoseLib')
import os
import PyOpenPose as OP
import time
import cv2
import numpy as np

from Image import Image

OPENPOSE_ROOT = "/home/simon/openpose"

def showHeatmaps(hm):
    for idx, h in enumerate(hm):
        cv2.imshow("HeatMap "+str(idx), h)


def showPAFs(PAFs, startIdx=0, endIdx=16):
    allpafs = []
    for idx in range(startIdx, endIdx):
        X = PAFs[idx*2]
        Y = PAFs[idx*2+1]
        tmp = np.dstack((X, Y, np.zeros_like(X)))
        allpafs.append(tmp)

    pafs = np.mean(allpafs, axis=0)
    cv2.imshow("PAF", pafs)


def getPosture(img):

	download_heatmaps = False

	# with_face = with_hands = False
	# op = OP.OpenPose((656, 368), (368, 368), (1280, 720), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
	#                  download_heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)
	
	op = OP.OpenPose((320, 240), (240, 240), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0, download_heatmaps)
 
	op.detectPose(img.image)
	print("POINTS")
	for thing in op.getKeypoints(op.KeypointType.POSE):
		print(thing)
	#op.detectFace(rgb)
	op.detectHands(img.image)

	res = op.render(img.image)
	persons = op.getKeypoints(op.KeypointType.HAND)[0]
	if download_heatmaps:
	    hm = op.getHeatmaps()
	    print("HM ", hm.shape, hm.dtype)
	    showHeatmaps(hm)

	    # hm = op.getHeatmaps()
	    # parts = hm[:18]
	    # background = hm[18]
	    # PAFs = hm[19:]  # each PAF has two channels (total 16 PAFs)
	    # cv2.imshow("Right Wrist", parts[4])
	    # cv2.imshow("background", background)
	    # showPAFs(PAFs)

	if persons is not None and len(persons) > 0:
	    print("First Person: ", persons[0].shape)

	cv2.imshow("OpenPose result", res)

	key = cv2.waitKey(0)
	exit()

if __name__ == '__main__':
	getPosture(Image(cv2.imread('imgsInDatabase/img1.jpg')))
