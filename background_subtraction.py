import argparse
import os
import sys
import shutil
import numpy as np
import json
import cv2
import cv2.cv as cv
from scipy.interpolate import UnivariateSpline

THRESHOLD = 45
MIN_AREA = 100
SCALE = 0.2
PADDING = 0.4


def create_LUT_8UC1(x, y):
    spl = UnivariateSpline(x, y)
    return spl(xrange(256))

def largestContour(contours):
    best_contour = None
    motion_found = False
    biggest_area = 0
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        # get an approximate area of the contour
        found_area = w*h 
        # find the largest bounding rectangle
        if (found_area > MIN_AREA) and (found_area > biggest_area):  
            biggest_area = found_area
            motion_found = True
            best_contour = (x, y, w, h)
    return motion_found, biggest_area, best_contour


incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
    [0, 70, 140, 210, 256])
decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
    [0, 30, 80, 120, 192])

def colder(frame):
    c_b, c_g, c_r = cv2.split(frame)
    c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
    return cv2.merge((c_b, c_g, c_r))

def filterImage(frame, base_frame):
    # resize
    try:
        frame = cv2.resize(frame, (0,0), fx=SCALE, fy=SCALE)
    except:
        print "frame shape " + frame.shape

    # colder
    frame = colder(frame)

    # blur
    frame = cv2.GaussianBlur( frame, ( 3, 3 ), 0)
    # frame = cv2.bilateralFilter( frame, -1, 50, 3)
    
    # background subtraction
    frameDelta = cv2.absdiff(base_frame, cv2.convertScaleAbs(frame))

    # grayscale
    frameDelta = cv2.cvtColor(frameDelta,cv2.COLOR_BGR2GRAY)

    # binary image
    frame = cv2.threshold(frameDelta, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    
    return frame


def createDirectory(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def getPaddedSquareImage(best_contour, frame):
    x, y, w, h = best_contour
    paddingY = h * PADDING
    paddingX = w * PADDING
    x = int(x - paddingX / 2.0)
    x = int(max(0, x))
    y = int(y - paddingY / 2.0)
    y = int(max(0, y))

    height, width, depth = frame.shape
    endX = int(x + w + paddingX)
    endX = min(endX, width)
    endY = int(y + h + paddingY)
    endY = min(endY, height)

    paddedWidth = endX - x
    paddedHeight = endY - y
    top, bottom, left, right, size = 0, 0, 0, 0, 0
    if paddedWidth > paddedHeight:
        top = (paddedWidth - paddedHeight) / 2
        bottom = top
        size = paddedWidth
    else:
        left = (paddedHeight - paddedWidth) / 2
        right = left
        size = paddedHeight
    roi = frame[y:endY, x:endX]

    square_roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return square_roi, {'x': x, 'y': y, 'top': top, 'left': left, 'size': size}

def main(args):
    createDirectory(args.outputDir)
    cap = cv2.VideoCapture(args.inputVideo) #./Media/vid/ryan_no_rope.MP4

    # fgbg = cv2.BackgroundSubtractorMOG()

    # first frame is the base for background subtraction
    ret, base_frame = cap.read()
    base_frame = cv2.resize(base_frame, (0,0), fx=SCALE, fy=SCALE)
    base_frame = cv2.GaussianBlur( base_frame, ( 3, 3 ), 0)

    framecount = 1
    data = {}
    while(1):
        ret, frame = cap.read()
        if frame is None:
            break

        framecount += 1
        if framecount % 10 == 0:
            print framecount
        
        frame_filtered = filterImage(frame, base_frame)
        # cv2.imshow('frame', frame_filtered)

        # fgmask = fgbg.apply(frame_filtered)
        # cv2.imshow('frame',fgmask)

        frame_filtered = cv2.dilate(frame_filtered, None, iterations=2)
        # cv2.imshow('frame',frame_filtered)

        contours, hierarchy = cv2.findContours(frame_filtered,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        frame = cv2.resize(frame, (0,0), fx=SCALE, fy=SCALE)
        # cv2.drawContours(frame, contours, -1, (0,0,255), 1)

        motion_found, biggest_area, best_contour = largestContour(contours)
        if motion_found:
            square_roi, transform = getPaddedSquareImage(best_contour, frame)
            cv2.imshow('frame', square_roi)
            cv2.imwrite(args.outputDir.strip('/') + '/' + str(framecount) + '.jpg', square_roi)
            data[framecount] = transform

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    with open(args.outputFile, 'w') as outfile:
        json.dump(data, outfile, indent=4, separators=(',', ': '))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputVideo", help="Filepath of the video to analyze", type=str)
    parser.add_argument("outputDir", help="Directory to store the square frames", type=str)
    parser.add_argument("outputFile", help="File to write frame info", type=str)
    args = parser.parse_args()
    if not os.path.isfile(args.inputVideo):
        print "ERROR: '" + args.inputVideo + "' does not exist or is not a file"
        exit(1)
    main(args)


