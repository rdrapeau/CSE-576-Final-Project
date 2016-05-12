import os
import sys
import glob

import dlib
from skimage import io

detector = dlib.simple_object_detector("detector.svm")
win_det = dlib.image_window()
win_det.set_image(detector)

dlib.hit_enter_to_continue()