import argparse
import os
import sys
import numpy as np
import json
import cv2
import cv2.cv as cv
import math
import numpy as np
from scipy import interpolate

SCALE = 0.2

def load_and_smooth_pose(pose_file, k):
	pose_data = open(pose_file).read()
	poses = json.loads(pose_data)

	num_photos = len(poses.keys())
	bounds = num_photos / k
	count = 1
	result = []
	while(count <= num_photos):
		img_name = str(count) + '.jpg'
		count = count + 1
		if img_name not in poses: continue
		summation = np.array(poses[img_name])
		norm_const = 1
		for j in range(k - 1):
			if count > num_photos: break
			img_name = str(count) + '.jpg'
			count = count + 1
			if img_name not in poses: continue
			summation = summation + np.array(poses[img_name])
			norm_const = norm_const + 1

		summation = summation / norm_const
		result.append(summation)

	result = np.matrix(result)


	interps = []
	# Build 32 interpolators (lol)
	for i in range(result.shape[1]):
		# ew..
		interps.append(interpolate.interp1d(np.arange(0, result.shape[0]), np.asarray(result[:, i].flatten())[0], kind='slinear'))

	return interps



def main(args):
	cap = cv2.VideoCapture(args.inputVideo)

	# video writer
	fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
	width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH) * SCALE)
	height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT) * SCALE)
	video = cv2.VideoWriter('video_viz.mp4',fourcc,30,(width,height))

	k = 4
	transform_data = open(args.transforms).read()
	transforms = json.loads(transform_data)
	poses = load_and_smooth_pose(args.poses, k)

	framecount = 0
	while(1):
		ret, frame = cap.read()
		if frame is None:
			break

		framecount += 1
		# if framecount % 10 == 0:
		#     print framecount

		if str(framecount) not in transforms:
			continue

		frame = cv2.resize(frame, (0,0), fx=SCALE, fy=SCALE)

		img_name = str(framecount) + '.jpg'
		img = cv2.imread(args.framesDir.strip('/') + '/' + img_name)
		transform = transforms[str(framecount)]
		pose = [f(framecount / 4.0) for f in poses]

		xPos = transform['x'] - transform['left']
		yPos = transform['y'] - transform['top']

		for i in range(0, len(pose), 2):
			x = pose[i]
			y = pose[i + 1]
			cv2.circle(frame, (xPos + int(x * transform['size'] * 4), yPos + int(y * transform['size']) * 4), 2, (0,0,255))

		cv2.imshow('frame', frame)
		video.write(frame)

		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

	cap.release()
	video.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputVideo", help="Filepath of the video to analyze", type=str)
    parser.add_argument("framesDir", help="Directory with the square frames", type=str)
    parser.add_argument("transforms", help="JSON file with transform info", type=str)
    parser.add_argument("poses", help="JSON file with pose info", type=str)
    args = parser.parse_args()
    if not os.path.isfile(args.inputVideo):
        print "ERROR: '" + args.inputVideo + "' does not exist or is not a file"
        exit(1)
    main(args)