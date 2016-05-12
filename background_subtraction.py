import numpy as np
import cv2
import cv2.cv as cv

cap = cv2.VideoCapture('./Media/vid/daniel_climbing_1_stabilized.mov')

fgbg = cv2.BackgroundSubtractorMOG()

fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH) * 0.2)
height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT) * 0.2)

video = cv2.VideoWriter('video.mp4',fourcc,24,(width,height))

framecount = 0
while(1):
    framecount += 1
    ret, frame = cap.read()
    
    if framecount % 10 == 0:
        print framecount
    
    frame = cv2.resize(frame, (0,0), fx=0.2, fy=0.2)

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)

    fgframe = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
    video.write(fgframe)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
video.release()
cv2.destroyAllWindows()
