import cv2
import numpy as np
import math

from matplotlib import pyplot as plt

PIXEL_THRESHOLD = 200

def in_bounds(img, row, col):
	height = len(img)
	width = len(img[0])
	return row >= 0 and row < height and col >= 0 and col < width


def check_neighbors(img, pix, visited):
	component = []
	for i in xrange(-1, 2):
		for j in xrange(-1, 2):
			t = (pix[0] + i, pix[1] + j)
			if t in img and t not in visited:
				component.append(t)
	return component


def connected_components(pixs):
	components = []
	visited = {}

	for k in pixs.keys():
		if k in visited:
			continue

		stack = [k]
		visited[k] = True

		# Go deeper
		component = []
		while len(stack) != 0:
			pixel = stack.pop()
			component.append(pixel)
			neighbors = check_neighbors(pixs, pixel, visited)
			for neighbor in neighbors:
				visited[neighbor] = True
				stack.append(neighbor);
		components.append(component)

	return components


def draw_components(components, filename):
	base = cv2.imread('Media/img/ryan_landscape_1.jpg', cv2.IMREAD_COLOR)
	for component in components:
		mean_x = np.mean([pixel[0] for pixel in component])
		mean_y = np.mean([pixel[1] for pixel in component])
		weight = min(len(component) / 30, 75)
		cv2.circle(base, (int(mean_y), int(mean_x)), weight, [255, 0, 0], thickness=-1)
	cv2.imwrite(filename, base)


def is_overlap(c1, c2):
	dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
	return dist < (c1[2] + c2[2])


def cluster(components):
	clusters = []
	for component in components:
		mean_x = np.mean([pixel[0] for pixel in component])
		mean_y = np.mean([pixel[1] for pixel in component])
		weight = min(len(component) / 10, 100)
		clusters.append((mean_x, mean_y, weight))

	new_clusters = []
	while len(clusters) > 0:
		new_cluster = clusters[0]
		cluster_list = [cluster for cluster in clusters if is_overlap(new_cluster, cluster)]
		total_weight = sum([cluster[2] for cluster in cluster_list])

		# calculate new cluster params
		new_cluster_x = 0.0
		new_cluster_y = 0.0
		new_cluster_weight = 0.0
		for cluster in cluster_list:
			new_cluster_x += cluster[0] * cluster[2] / total_weight
			new_cluster_y += cluster[1] * cluster[2] / total_weight
			new_cluster_weight += cluster[2]
			clusters.remove(cluster)

		if new_cluster_weight > 20:
			new_clusters.append((new_cluster_x, new_cluster_y, new_cluster_weight))

	return new_clusters


def draw_clusters(clusters, filename):
	base = cv2.imread('Media/img/ryan_landscape_1.jpg', cv2.IMREAD_COLOR)
	for cluster in clusters:
		cv2.circle(base, (int(cluster[1]), int(cluster[0])), int(math.log(cluster[2]) * 5), [0, 255, 0], thickness=-1)

	cv2.imwrite(filename, base)