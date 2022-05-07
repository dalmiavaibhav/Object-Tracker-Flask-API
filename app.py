from flask import Flask, render_template, Response, request, redirect, url_for
from flask_socketio import SocketIO
from ObjectTracking.siamfc import ops
import matplotlib.pyplot as plt
import ObjectDetection
import ObjectTracking
import numpy as np
import cv2
import base64
import json
import os

app = Flask(__name__)
socketio = SocketIO(app)

SOT = ['SIAMFC', 'DLIB', 'Goturn', 'CSRT', 'MIL', 'BOOSTING', 'MOSSE', 'KCF', 'TLD', 'MEDIANFLOW']
MOT = ['SORT', 'DEEPSORT']

vs = None
detector = None
tracker = None
tracker_name = None
tracks = None
vs_name = None
gt = None
f = None

def clean_resources():
    delete_vs()
    global vs, detector, tracker, tracker_name, tracks, vs_name, gt, f
    vs = None
    detector = None
    tracker = None
    tracker_name = None
    tracks = None
    vs_name = None
    gt = None
    f = None

def plot_graph(x, y, filename, xlabel='Overlap threshold', ylabel='Succes rate'):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.savefig(f'{filename}_overlap.jpeg')
    #plt.show()

def success_plot(tracks, gt):
    iou = ops.intersection_over_union(tracks, gt) 
    x = np.linspace(0,1,100)
    y = np.zeros(100)
    for i,thresh in enumerate(x):
        iou_ = iou[np.where(iou>thresh)]
        success_rate = (iou_.shape[0]/gt.shape[0])
        y[i] = success_rate

    return (x, y)

def precision_plot(tracks, gt):
    eu = ops.eucledian_distance(tracks, gt)
    x = np.linspace(0,50,51)
    y = np.zeros(51)
   
    for i,thresh in enumerate(x):
        eu_ = eu[np.where(eu<thresh)]
        precision_rate = (eu_.shape[0]/gt.shape[0])
        y[i] = precision_rate
        
    return (x,y)

def load_gt(vs_name):
    try:
        gt_path=f'./static/annotations/{vs_name}.txt'
        annotations = np.loadtxt(gt_path, delimiter=',')
        print('gt loaded.')
        return annotations
    except:
        print('Annotation file not found')
        return None

def eval(tracks, vs_name):
    gt_path=f'./static/annotations/{vs_name}.txt'
    try:
        annotations = np.loadtxt(gt_path, delimiter=',')
        annotations = np.resize(annotations, tracks.shape)
        (x1,y1) = success_plot(tracks, annotations)
        #plot_graph(x,y, vs_name)
        (x2,y2) = precision_plot(tracks, annotations)
        #plot_graph(x,y, vs_name)
        return (x1,y1,x2,y2)
    except:
        return None

def convert_image_to_jpeg(image):
        # Encode frame as jpeg
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # Encode frame in base64 representation and remove
        # utf-8 encoding
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)

def gen():
    global vs, detector, tracker, SOT, tracker_name, MOT, tracks, gt, f
    
    while True:    
        f+=1
        ret, frame = vs.read()
        if not ret:
            print("Error in receiving frame.")
            break
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timer = cv2.getTickCount()
        
        if gt is not None:
            cv2.rectangle(frame, (int(gt[f][0]), int(gt[f][1])), (int(gt[f][0]+gt[f][2]), int(gt[f][1]+gt[f][3])), (0, 255, 0), 2, 1)
           
        if tracker_name in SOT:
            #print('SOT')
            trackers = tracker.track(frame)
            #print(f'trackers {trackers} tracks {tracks}')
            
        if tracker_name in MOT:
            #print('MOT')
            dets = detector.detect(frame)
            trackers = tracker.track(dets)
            
        fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
        #print(f'fps:{fps}')
        #cv2.putText(frame, f'fps:{fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if trackers is not None:
            if tracker_name in SOT:
                box = np.array(trackers[0].tolist())
                box[2:] -= box[:2]
                tracks.append(box.tolist())
            
            for track in trackers:
                cv2.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), (0, 0, 255), 2, 1)
                if len(track) > 4:
                    cv2.putText(frame, f'id:{track[4]}', (int(track[0]), int(track[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                else:
                    cv2.putText(frame, f'id:1', (int(track[0]), int(track[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
    
@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def connection():
    print('A new client connected.')

@socketio.on('disconnect')
def disconnect():
    clean_resources()
    print('Client Disconnected.')

@socketio.on('create_vs')
def create_vs(input):
    try:
        global vs, vs_name, f
        f = -1
        vs_name = input
        if input == 'webcam':
            vs = cv2.VideoCapture(0)
        else:
            vs = cv2.VideoCapture(os.path.join('static', input))

        print('Video Source Created.')
    except:
        print('Could not open video source.')

@socketio.on('delete_vs')
def delete_vs():
    global vs
    try:
        vs.release()
    except:
        pass
    print('Video Source Deleted.')

@socketio.on('det')
def create_detector(model):
    global detector
    if model == 'MobileNetSSD':
        detector = ObjectDetection.MobileNetSSD.MobileNetSSD()
    if model == 'Yolo':
        detector = ObjectDetection.Yolo.Yolo()
    print('Model Loaded.')

@socketio.on('init_tracker')
def init_tracker():
    global vs, f
    f+=1
    ret, frame = vs.read()
    if not ret:
        print("Error in receiving frame.")
    socketio.emit('init_frame', { 'image': convert_image_to_jpeg(frame),
                                    'size': (frame.shape[0], frame.shape[1]) })
    print('Init frame emitted')

@socketio.on('trk')
def load_tracker(name, box):
    global f, tracker, vs, tracker_name, SOT, MOT, tracks, gt
    if name in SOT:
        tracker_name = name
        tracks = []
        gt = load_gt(vs_name[:-4])
        print(box)
        f+=1
        ret, frame = vs.read()
        if tracker_name == 'SIAMFC':
            tracker = ObjectTracking.siamfc.TrackerSiamFC(frame, box)
        if tracker_name == 'Goturn':
            tracker = ObjectTracking.goturn.Goturn(frame, box)
        if tracker_name == 'CSRT':
            tracker = ObjectTracking.csrt.CSRT(frame, box)
        if tracker_name == 'MIL':
            tracker = ObjectTracking.mil.MIL(frame, box)
        if tracker_name == 'BOOSTING':
            tracker = ObjectTracking.boosting.BOOSTING(frame, box)
        if tracker_name == 'MEDIANFLOW':
            tracker = ObjectTracking.medianflow.MEDIANFLOW(frame, box)
        if tracker_name == 'KCF':
            tracker = ObjectTracking.kcf.KCF(frame, box)
        if tracker_name == 'TLD':
            tracker = ObjectTracking.tld.TLD(frame, box)
        if tracker_name == 'MOSSE':
            tracker = ObjectTracking.mosse.MOSSE(frame, box)
        if tracker_name == 'DLIB': 
            tracker = ObjectTracking.DLIB.DLIB(frame, box)
    if name in MOT:
        tracker_name = name
        if tracker_name == 'SORT':
            tracker = ObjectTracking.sort.Sort()
        if tracker_name == 'DEEPSORT':
            pass    
    print('tracker loaded')

@socketio.on('start')
def start_tracking():
    url = url_for('video_feed')
    socketio.emit('vs', url)
    print('Tracking Started')

@socketio.on('stop')
def stop_tracking():
    global is_running, tracks, vs_name, gt
    if gt is not None:
        (x1, y1, x2, y2) = eval(np.array(tracks), vs_name[:-4])
        x1 = json.dumps(x1.tolist())
        y1 = json.dumps(y1.tolist())
        x2 = json.dumps(x2.tolist())
        y2 = json.dumps(y2.tolist())
        socketio.emit('plot_graphs',  {'x': [x1,x2] , 'y': [y1,y2]} )
        print('Plot Graphs')
    print('Tracking Stopped')
    clean_resources()

if __name__ == '__main__':
    try:
        socketio.run(app=app, host='0.0.0.0', port=5000)
    except Exception as e:
        print(e)
