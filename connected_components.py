import cv2
import numpy as np

from matplotlib import pyplot as plt

PIXEL_THRESHOLD = 200

def in_bounds(img, row, col):
	height = len(img)
	width = len(img[0])
	return row >= 0 and row < height and col >= 0 and col < width


def check_neighbors(img, row, col):
	component = []
	for i in xrange(-1, 2):
		for j in xrange(-1, 2):
			if in_bounds(img, row + i, col + j) and img[row + i][col + j] > PIXEL_THRESHOLD:
				img[row + i][col + j] = 0
				component.append(tuple([row + i, col + j]))

	return component


def connected_components(img):
	components = []
	for row in xrange(len(img)):
		for col in xrange(len(img[0])):
			if img[row][col] > PIXEL_THRESHOLD:
				stack = [tuple([row, col])]

				# Go deeper
				component = []
				while len(stack) != 0:
					pixel = stack.pop()
					component.append(pixel)
					stack += check_neighbors(img, pixel[0], pixel[1])

				components.append(component)

	return components


img = cv2.imread('out_bigger_window.jpg', cv2.IMREAD_GRAYSCALE)
components = connected_components(img)

base = cv2.imread('Media/img/ryan_landscape_1.jpg', cv2.IMREAD_COLOR)

# Filter
components = [component for component in components if len(component) > 75]

for component in components:
	mean_x = np.mean([pixel[0] for pixel in component])
	mean_y = np.mean([pixel[1] for pixel in component])
	cv2.circle(base, (int(mean_y), int(mean_x)), 20, [255, 0, 0], thickness=-1)

cv2.imwrite('image.jpg', base)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
