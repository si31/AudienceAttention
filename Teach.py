import sys
import cv2
import numpy as np
from Person import Person
from Image import Image
import Main

def compareLabels():
	img = Main.readFromDatabase(sys.argv[1])
	comparisons = []
	for person in img.persons[0:2]:
		person.accumulateData()
		label = person.labels[0]
		thisComparison = ([person.poseArea, person.occlusion, person.postureArea, person.blur], [label.humanPoseAngle, label.humanOcclusion, label.humanPostureLR, label.humanMovement])
		comparisons.append(thisComparison)
		print(thisComparison)


def main():
	compareLabels()

if __name__ == "__main__":
	main()
