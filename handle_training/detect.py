import os
import sys
import glob

import dlib
from skimage import io
from skimage import img_as_ubyte
from skimage import color

if len(sys.argv) < 2:
    print(
        "Give the path to the examples/faces directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./detect.py ../examples/faces [-s]")
    exit()
f = sys.argv[1]

s = False
if len(sys.argv) > 2 and sys.argv[2] == '-s':
    s = True

detector = dlib.simple_object_detector("detector.svm")

print("Processing file: {}".format(f))
img = io.imread(f)
img = color.rgb2gray(img)
img = img_as_ubyte(img)
dets = detector(img)
print("Number of detected: {}".format(len(dets)))
for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))

if s:
    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()
