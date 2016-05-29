#!/usr/bin/python
import argparse
import os
import sys
import numpy as np
import json
import cv2
import cv2.cv as cv
import math
import numpy as np
import heatmap
from scipy import interpolate
from scipy.spatial.distance import euclidean


SCALE = 0.2
MEAN_MAD_THRESHOLD = 2.10904
HM_DECAY = 0.90


# def bin_number_freedman_diaconis(data):
#     std = np.std(data)
#     bin_size = 2 * std * (len(data) ** (-1.0 / 3.0))
#     number_of_bins = math.ceil((np.max(data) - np.min(data)) / bin_size)
#     return bin_size, number_of_bins


def median_absolute_deviation(poses, t):
	img_names = ['%d.jpg' % i for i in xrange(1, len(poses) + 1) if '%d.jpg' % i in poses]
	poses = {img : np.array(poses[img]) for img in poses}
	window_start_indexes = [i for i in xrange(t + 1, len(img_names) - 1, 2 * t + 1)]
	resulting_poses = [poses[img_names[0]]]
	time_steps = [0]
	for i in window_start_indexes:
		img_name = img_names[i]
		if i < t or i > len(img_names) - t - 1:
			continue

		img_names_window = img_names[i - t : i] + [img_name] + img_names[i + 1 : i + t + 1]
		poses_window = [poses[img] for img in img_names_window]
		current_pose = poses[img_name]

		pose_distances = np.array([euclidean(current_pose, other_pose) for other_pose in poses_window])
		median_distance = np.median(pose_distances)

		deviations = [abs(median_distance - pose_distance) for pose_distance in pose_distances]
		median_deviation = np.median(deviations)

		pose_distances = np.abs(pose_distances - median_distance) / median_deviation
		for j, distance in enumerate(pose_distances):
			if distance < MEAN_MAD_THRESHOLD:
				time_steps.append(i - t + j)
				resulting_poses.append(poses_window[j])

	resulting_poses.append(poses[img_names[-1]])
	time_steps.append(len(img_names) - 1)
	return resulting_poses, np.array(time_steps)


def load_and_smooth_pose(pose_file, k):
	pose_data = open(pose_file).read()
	poses = json.loads(pose_data)

	num_photos = len(poses.keys())
	bounds = num_photos / k
	count = 1
	result, time_steps = median_absolute_deviation(poses, k)
	smoothed_result = []
	for i, pose in enumerate(result):
		summation = pose
		norm_const = 1
		for j in xrange(k - 1):
			if count > num_photos: break
			count += 1
			summation += result[i + j]
			norm_const += 1

		smoothed_result.append(summation / norm_const)

	smoothed_result = np.matrix(smoothed_result)
	interps = []
	# Build 32 interpolators (lol)
	for i in range(smoothed_result.shape[1]):
		# ew..
		interps.append(interpolate.interp1d(time_steps, np.asarray(smoothed_result[:, i].flatten())[0], kind='slinear'))

	return interps



def main(args):
	cap = cv2.VideoCapture(args.inputVideo)

	# video writer
	fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
	width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH) * SCALE)
	height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT) * SCALE)
	video = cv2.VideoWriter('video_viz.mp4',fourcc,30,(width,height))

	k = 10
	transform_data = open(args.transforms).read()
	transforms = json.loads(transform_data)
	poses = load_and_smooth_pose(args.poses, k)
	hm = None
	hm_gaussian = heatmap.gaussian_template(15, 4)

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
		if hm is None:
			hm = heatmap.new_heatmap(frame.shape[0], frame.shape[1])

		transform = transforms[str(framecount)]
		pose = [f(framecount) for f in poses]
		xPos = transform['x'] - transform['left']
		yPos = transform['y'] - transform['top']

		img_name = str(framecount) + '.jpg'
		img = cv2.imread(args.framesDir.strip('/') + '/' + img_name)
		pose = [f(framecount) for f in poses]
		for i in range(0, len(pose), 2):
			pose[i] = int(pose[i] * transform['size'] * 4) + xPos
			pose[i + 1] = int(pose[i + 1] * transform['size'] * 4) + yPos

		heatmap.update_heatmap(hm, hm_gaussian, pose, HM_DECAY)

		for i in range(0, len(pose), 2):
			x = pose[i]
			y = pose[i + 1]
			cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))

		cv2.imshow('hm', hm)
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
