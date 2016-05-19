import sys
import numpy as np
import cv2
import cv2.cv as cv

def filterImage(frame, width, height):
    frame = cv2.resize(frame, (0,0), fx=0.2, fy=0.2)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur( frame, ( 3, 3 ), 0)
    return frame

cap = cv2.VideoCapture('./Media/vid/ryan_no_rope.MP4')

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
    
    frame_filtered = filterImage(frame, width, height)# cv2.resize(frame, (0,0), fx=0.2, fy=0.2)

    fgmask = fgbg.apply(frame_filtered)
    # cv2.imshow('frame',fgmask)

    contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    frame = cv2.resize(frame, (0,0), fx=0.2, fy=0.2)
    cv2.drawContours(frame, contours, -1, (0,0,255), 2)

    # flatten
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    if len(boundingBoxes) > 0:
        minX = sys.maxint
        maxX = -sys.maxint - 1
        minY = sys.maxint
        maxY = -sys.maxint - 1
        for bb in boundingBoxes:
            # [item for sublist in contours for item in sublist]
            # contours = np.array(contours)
            x,y,w,h = bb
            if (x < minX):
                minX = x
            if (x + w > maxX):
                maxX = x + w
            if (y < minY):
                minY = y
            if (y + h > maxY):
                maxY = y + h
        cv2.rectangle(frame,(minX,minY),(minX + (maxX - minX), minY + (maxY - minY)),(0,255,0),1)

    cv2.imshow('contours',frame)


    # fgframe = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
    video.write(frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
video.release()
cv2.destroyAllWindows()


