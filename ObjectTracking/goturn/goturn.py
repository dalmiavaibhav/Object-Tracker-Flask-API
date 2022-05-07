import cv2
import numpy as np

class Goturn:
    def __init__(self, frame, bbox=[600, 296, 101, 107]):
        self.tracker = cv2.TrackerGOTURN_create()
        #bbox = cv2.selectROI('frame', frame, showCrosshair=False)
        #bbox = [x, y, w, h]
        print(bbox)
        self.tracker.init(frame, bbox)
        
    def track(self,frame):
        
            timer = cv2.getTickCount()

            ok, bbox = self.tracker.update(frame)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            #cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
            #cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            #print(f'fps: {fps}')
            if ok:
                box = np.array(bbox)
                box[2:] += box[:2]
                return np.expand_dims(box, axis=0)
            else:
                return None
                #cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                #print(f'tracking failure detected')
""" 
if __name__ == "__main__":

    vs = cv2.VideoCapture('../../video2.mp4')
    ret, frame = vs.read()

    tracker = Goturn(frame)

    while True:
        ret, frame = vs.read()

        tracks = tracker.track(frame)

        if tracks is not None:
            cv2.rectangle(frame, (int(tracks[0]), int(tracks[1])), (int(tracks[2]), int(tracks[3])), (0, 255, 0), 2, 1 ) 

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    vs.release() 
"""