#Import the neccesary libraries
import numpy as np
import argparse
import cv2 
    
class MobileNetSSD:
    def __init__(self , prototxt='./ObjectDetection/MobileNetSSD_deploy.prototxt', weights='./ObjectDetection/MobileNetSSD_deploy.caffemodel', thr=0.2):
        # construct the argument parse 
        self.config = {}
        self.config['prototxt'] = prototxt
        self.config['weights'] = weights
        self.config['thr'] = thr
        
        self.classNames = { 0: 'background',
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

        #Load the Caffe model 
        self.net = cv2.dnn.readNetFromCaffe(self.config['prototxt'], self.config['weights'])

    def detect(self, frame):
    
        frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size. 
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        self.net.setInput(blob)
        #Prediction of network
        detections = self.net.forward()

        #Size of frame resize (300x300)
        cols = frame_resized.shape[1] 
        rows = frame_resized.shape[0]

        #For get the class and location of object detected, 
        # There is a fix index for class, location and confidence
        # value in @detections array .
        dets = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] #Confidence of prediction 
            if confidence > self.config['thr']: # Filter prediction 
                class_id = int(detections[0, 0, i, 1]) # Class label

                # Object location 
                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                
                # Factor for scale to original size of frame
                heightFactor = frame.shape[0]/300.0  
                widthFactor = frame.shape[1]/300.0 
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
                # Draw location of object  
                
                dets.append([xLeftBottom, yLeftBottom,xRightTop,yRightTop])
                
                #cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                #            (0, 255, 0), 2, 1)

                # Draw label and confidence of prediction in frame resized
                """ if class_id in self.classNames:
                    label = self.classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                        (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                        (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                    print(label) #print class and confidence 
                """
        return np.array(dets)

