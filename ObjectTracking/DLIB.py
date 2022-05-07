import numpy as np
import dlib
import cv2

class DLIB:
	def __init__(self, frame, box=[0, 0, 200, 200]):
		self.tracker = dlib.correlation_tracker()
		box = np.array(box)
		box[2:] += box[:2]
		(startX, startY, endX, endY) =box.astype("int")
		rect = dlib.rectangle(startX, startY, endX, endY)
		print(rect)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		self.tracker.start_track(rgb, rect)

	def track(self, frame):
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)	
		self.tracker.update(rgb)
		pos = self.tracker.get_position()
		
		if pos is not None:
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
			box = np.array([startX, startY, endX, endY])
			return np.expand_dims(box, axis=0)
		else:
			return None
	
"""	
if __name__ == "__main__":

	vs = cv2.VideoCapture('video2.mp4')
	ret, frame = vs.read()
	tracker = DLIB(frame)

    
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