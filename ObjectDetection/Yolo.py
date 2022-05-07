# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html
# Usage example:  python3 object_detection_yolo.py --video=run.mp4 --device 'cpu'
#                 python3 object_detection_yolo.py --video=run.mp4 --device 'gpu'
#                 python3 object_detection_yolo.py --image=bird.jpg --device 'cpu'
#                 python3 object_detection_yolo.py --image=bird.jpg --device 'gpu'
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

class Yolo:
    def __init__(self, confThreshold = 0.5, nmsThreshold = 0.4, inpWidth = 416, inpHeight = 416):
        # Initialize the parameters
        self.confThreshold = confThreshold  #Confidence threshold
        self.nmsThreshold = nmsThreshold   #Non-maximum suppression threshold
        self.inpWidth = inpWidth        #Width of network's input image
        self.inpHeight = inpHeight      #Height of network's input image

        self.config = {}


        self.config['device'] = 'cpu'
        self.config['image'] = ''
        self.config['video'] = ''
                
        # Load names of classes
        classesFile = "./ObjectDetection/coco.names"
        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        # Give the configuration and weight files for the model and load the network using them.
        self.modelConfiguration = "./ObjectDetection/yolov3.cfg"
        self.Weights = "./ObjectDetection/yolov3.weights"

        self.net = cv.dnn.readNetFromDarknet(self.modelConfiguration, self.Weights)

        if(self.config['device'] == 'cpu'):
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            print('Using CPU device.')
        elif(self.config['device'] == 'gpu'):
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
            print('Using GPU device.')

        # Get the names of the output layers
    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        
        label = '%.2f' % conf
            
        # Get the label for the class name and its confidence
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        dets = []
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            dets.append([left, top, left + width, top + height])
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            #self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return np.array(dets)


    def detect(self, frame):
        #frame_resized = cv.resize(frame,(self.inpWidth, self.inpHeight)) # resize frame for prediction
        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.getOutputsNames())

        # Remove the bounding boxes with low confidence
        dets = self.postprocess(frame, outs)

        return dets

""" if __name__ == '__main__': 
    model = Yolo() 
    vs = cv.VideoCapture(0) 
    while True: 
        ret, frame = vs.read() 
        dets = model.detect(frame) 
        print(dets)
        for track in dets:
            cv.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), (0, 255, 0), 2, 1)
            #cv.putText(frame, f'id:{track[4]}', (int(track[0]), int(track[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv.imshow('frame', frame) 
        if cv.waitKey(1) == ord('q'): 
            break 
    
    vs.release()  """