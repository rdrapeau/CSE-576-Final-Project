import numpy as np
import cv2
import cv2.cv as cv

def run_main():
    cap = cv2.VideoCapture('./Media/vid/ryan_no_rope.MP4')

    # video writer
    fourcc = cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
    width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    video = cv2.VideoWriter('video_ryan_tracking.mp4',fourcc,30,(width,height))

    # Skip forward 150 frames
    for i in range(150):
        ret, frame = cap.read()

    # Grab the next frame as starting point
    ret, frame = cap.read()
    cv2.imshow('Tracking', frame)

    # Set the ROI (Region of Interest). Actually, this is a
    # rectangle of the building that we're tracking
    c,r,w,h = 925,280,200,400
    track_window = (c,r,w,h)
    cv2.rectangle(frame, (c,r), (c+w,r+h), 255, 2)
    
    cv2.imwrite('test.png', frame)

    # Create mask and normalized histogram
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 30.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)
    
    while True:
        ret, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        x,y,w,h = track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
        
        cv2.imshow('Tracking', frame)
        video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_main()