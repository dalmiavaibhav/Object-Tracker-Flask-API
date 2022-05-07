import cv2
import numpy as np

class MIL:
    def __init__(self, frame, bbox=[0, 0, 200, 200]):
        self.tracker = cv2.TrackerMIL_create()
        self.tracker.init(frame, bbox)
    def track(self, frame):
        ok, bbox = self.tracker.update(frame)

        if ok:
            box = np.array(bbox)
            box[2:] += box[:2]
            return np.expand_dims(box, axis=0)
        else:
            return None
"""
if __name__ == '__main__':
    
    vs = cv2.VideoCapture('../../../video2.mp4')
    ret, frame = vs.read()

    tracker = MIL(frame)

    while True:
        ret, frame = vs.read()

        tracks = tracker.track(frame)

        if tracks is not None:
            for track in tracks:
                cv2.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), (0, 255, 0), 2, 1 ) 

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    vs.release()  
"""