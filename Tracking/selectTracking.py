import cv2
import time

file_name = './onlyTracking.py/../Video/onTheRoad_01.mp4'
frame_count = 0

color = (0, 0, 255)

trackers = cv2.legacy.MultiTracker()

cap = cv2.VideoCapture(file_name)


while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 0)
    frameNew = cv2.resize(
        frame, (round(0.6*frame.shape[0]), round(0.6*frame.shape[1])))

    (height, width) = frameNew.shape[:2]
    print()
    print(width, height)
    print(' x , y ', end='')
    (rangeX , rangeY , w , h ) = ( width // 2 ,  height // 2 , width//10 , height//10)
    (startX , startY , endX , endY) = (rangeX- w , rangeY -h , rangeX+ w , rangeY + h)
    # cv2.circle(frameNew, (x, y), 10, color, -1)

    cv2.rectangle(frameNew , (startX,startY) , (endX , endY) , color , 3)

    if frameNew is None:
        print('### No more frame ###')
        break

    start_time = time.time()
    frame_count += 1
    (success, boxes) = trackers.update(frameNew)

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frameNew, (x, y), (x + w, y + h), (0,255,0), 2)
        if  (startX < x and startY < y) and (x + w < endX and y+h < endY):
             color = (255 , 0 , 0)
        else:
            color = (0 , 0 ,255)
    

    cv2.imshow("Frame", frameNew)
    frame_time = time.time() - start_time
    print("Frame {} time {}".format(frame_count, frame_time))

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        box = cv2.selectROI("Frame", frameNew, fromCenter=False,
                            showCrosshair=True)  # 십자모양

        tracker = cv2.legacy.TrackerCSRT_create()
        # tracker = cv2.legacy.TrackerKCF_create()

        # tracker = cv2.legacy.TrackerBoosting_create()
        # mil
        # tracker = cv2.legacy.TrackerMIL_create()
        # tld
        # tracker = cv2.legacy.TrackerTLD_create()
        # medianflow
        # tracker = cv2.TrackerMedianFlow_create()
        # mosse
        # tracker = cv2.TrackerMOSSE_create()
        trackers.add(tracker, frameNew, box)

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
