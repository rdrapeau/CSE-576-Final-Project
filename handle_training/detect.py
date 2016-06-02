import os
import sys
import glob

import dlib
from skimage import io
from skimage import img_as_ubyte
from skimage import color

def handle_locations(img_fp, detector_fp):
    detector = dlib.simple_object_detector(detector_fp)
    img = io.imread(img_fp)
    img = color.rgb2gray(img)
    img = img_as_ubyte(img)
    dets = detector(img)
    return dets


def main():
    if len(sys.argv) < 3:
        print(
            "Give the path to the examples/faces directory as the argument to this "
            "program. For example, if you are in the python_examples folder then "
            "execute this program by running:\n"
            "    ./detect.py ../examples/faces ./detector.svm [-s]")
        exit()

    f = sys.argv[1]
    detector_fp = sys.argv[2]

    s = False
    if len(sys.argv) > 3 and sys.argv[3] == '-s':
        s = True

    dets = handle_locations(f, detector_fp)
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

if __name__ == "__main__":
    main()

