import cv2
import time

file_name = './faceOnlyTracking.py/../ai_cv/video/face_01.mp4'
frame_count = 0

# tracker = cv2.legacy.TrackerCSRT_create()
tracker = cv2.legacy.TrackerKCF_create()
# boosting
# tracker = cv2.legacy.TrackerBoosting_create()
# tracker = cv2.legacy.TrackerMIL_create()
# tracker = cv2.legacy.TrackerTLD_create()
# tracker = cv2.legacy.TrackerMedianFlow_create()
# tracker = cv2.legacy.TrackerMOSSE_create()


face_cascade_name = './faceOnlyTracking.py/../ai_cv/haarcascades/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('### Error loading face cascade ###')
    exit(0)

detected = False
frame_mode = 'Tracking'
elapsed_time = 0
trackers = cv2.legacy.MultiTracker_create()

vs = cv2.VideoCapture(0)

while True:
        ret, frame = vs.read()
        if frame is None:
            print('###끝끝끝끝끝끝끝###')
            break
        start_time = time.time()
        frame_count += 1
        if detected:
            frame_mode = 'Tracking'
            (success, boxes) = trackers.update(frame)
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            frame_mode = 'Detection'
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)            
            faces = face_cascade.detectMultiScale(frame_gray)
            for (x,y,w,h) in faces:
               
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 4)
            # print(faces)
            # print(faces[0])
            trackers.add(tracker, frame, tuple(faces[0])) 
            detected = True

        
        cv2.imshow("Frame", frame)
        frame_time = time.time() - start_time
        elapsed_time += frame_time
        print("[{}] Frame {} time {}".format(frame_mode, frame_count, frame_time))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

print("Elapsed time {}".format(elapsed_time))
vs.release()
cv2.destroyAllWindows()
