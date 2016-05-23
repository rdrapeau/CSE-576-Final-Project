# import the necessary packages
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to a floating point data type
image = cv2.imread(args["image"])


# apply SLIC and extract (approximately) the supplied number
# of segments
res = cv2.pyrMeanShiftFiltering(image, 10, 20, 3);

# show the output of SLIC
fig = plt.figure("segments")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(res)
plt.axis("off")

# show the plots
plt.show()