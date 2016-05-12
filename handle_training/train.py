import os
import sys
import glob

import dlib
from skimage import io

if len(sys.argv) != 2:
    print(
        "Give the path to the examples/faces directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train.py ../examples/faces")
    exit()
folder = sys.argv[1]

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = False
options.C = 5
options.num_threads = 4
options.epsilon = 0.01
options.be_verbose = True


training_xml_path = os.path.join(folder, "train.xml")
# testing_xml_path = os.path.join(folder, "testing.xml")
dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)

print("")
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
# print("Testing accuracy: {}".format(
#     dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))