import cv2
from imutils.video import VideoStream
import imutils

vs = cv2.VideoCapture(0)
bs = cv2.createBackgroundSubtractorKNN(detectShadows = True)
firstFrame = None
while True:
    ret, frame = vs.read()
    text = "No Detection"
    if frame is None:
        break
    frame = imutils.resize(frame, width = 500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray , (27, 27), 0)
    if firstFrame is None:
        firstFrame = gray
        continue
    frameDel = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDel, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        text = "DETECTED"

    cv2.putText(frame, "status:{}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
	    break

cv2.destroyAllWindows()
