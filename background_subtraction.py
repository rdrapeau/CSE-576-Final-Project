import sys
import numpy as np
import cv2
import cv2.cv as cv
from scipy.interpolate import UnivariateSpline

THRESHOLD = 45
MIN_AREA = 100
SCALE = 0.2
 
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

def filterImage(frame, width, height, base_frame):
    # resize
    frame = cv2.resize(frame, (0,0), fx=SCALE, fy=SCALE)

    # colder
    c_b, c_g, c_r = cv2.split(frame)
    c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
    frame = cv2.merge((c_b, c_g, c_r))

    # blur
    frame = cv2.GaussianBlur( frame, ( 3, 3 ), 0)
    # frame = cv2.bilateralFilter( frame, -1, 50, 3)
   
    frameDelta = cv2.absdiff(base_frame, cv2.convertScaleAbs(frame))

     # grayscale
    frameDelta = cv2.cvtColor(frameDelta,cv2.COLOR_BGR2GRAY)

    frame = cv2.threshold(frameDelta, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    
    return frame

cap = cv2.VideoCapture('./Media/vid/ryan_no_rope.MP4')

fgbg = cv2.BackgroundSubtractorMOG()

fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH) * SCALE)
height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT) * SCALE)
video = cv2.VideoWriter('video.mp4',fourcc,24,(width,height))

ret, base_frame = cap.read()
base_frame = cv2.resize(base_frame, (0,0), fx=SCALE, fy=SCALE)
base_frame = cv2.GaussianBlur( base_frame, ( 3, 3 ), 0)

framecount = 1

while(1):
    framecount += 1
    ret, frame = cap.read()
    
    if framecount % 10 == 0:
        print framecount
    
    frame_filtered = filterImage(frame, width, height, base_frame)
    # cv2.imshow('frame', frame_filtered)

    # fgmask = fgbg.apply(frame_filtered)
    # cv2.imshow('frame',fgmask)

    frame_filtered = cv2.dilate(frame_filtered, None, iterations=2)
    # cv2.imshow('frame',frame_filtered)

    contours, hierarchy = cv2.findContours(frame_filtered,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    frame = cv2.resize(frame, (0,0), fx=SCALE, fy=SCALE)
    cv2.drawContours(frame, contours, -1, (0,0,255), 1)

    motion_found, biggest_area, best_contour = largestContour(contours)
    if motion_found:
        x, y, w, h = best_contour
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        roi_gray = frame_gray[y:y+h, x:x+w]
        cv2.imshow('frame',roi_gray)
        cv2.rectangle(frame,(x,y),(x + w, y + h),(0,255,0),1)
    # cv2.imshow('frame',frame)

    # cv2.imshow('contours',frame)


    # fgframe = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
    video.write(frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
video.release()
cv2.destroyAllWindows()


