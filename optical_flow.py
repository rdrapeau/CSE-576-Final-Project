import cv2
import numpy as np
import cv2.cv as cv

SCALE = 0.2

cap = cv2.VideoCapture('./Media/vid/ryan_no_rope.MP4')

# video writer
fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH) * SCALE)
height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT) * SCALE)
video = cv2.VideoWriter('video_opt.mp4',fourcc,30,(width,height))


# Note: questionable
def filterImage(frame):
    frame = cv2.resize(frame, (0,0), fx=SCALE, fy=SCALE)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    mean, stddev = cv2.meanStdDev(frame)
    frame = cv2.subtract(frame, mean)
    frame = cv2.normalize(frame,None,0,255,cv2.NORM_MINMAX)
    return frame

ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, (0,0), fx=SCALE, fy=SCALE)
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255


framecount = 0
while(1):
    framecount += 1
    if framecount % 10 == 0:
        print framecount

    ret, frame2 = cap.read()
    frame2 = cv2.resize(frame2, (0,0), fx=SCALE, fy=SCALE) #filterImage(frame2)
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    # prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow]
    flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 8, 15, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    cv2.imshow('frame2',rgb)
    video.write(rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

cap.release()
video.release()
cv2.destroyAllWindows()
