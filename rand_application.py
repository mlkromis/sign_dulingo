"""
Demo Flask application to test the operation of Flask with socket.io
Aim is to create a webpage that is constantly updated with random numbers from a background python process.
30th May 2014
===================
Updated 13th April 2018
+ Upgraded code to Python 3
+ Used Python3 SocketIO implementation
+ Updated CDN Javascript and CSS sources
"""

# Start with a basic flask app webpage.
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, url_for, copy_current_request_context, Response
from random import random
from time import sleep
from threading import Thread, Event, Lock
import cv2
import requests
from processing import square_pad, preprocess_for_vgg
from model import create_model
import argparse
import numpy as np
import string

# Map model names to classes
MODELS = ["resnet", "vgg16", "inception", "xception", "mobilenet"]


class VideoController:
    """
    Class that continuously gets and holds frames from a VideoCapture object
    """
    def __init__(self):
        self.stream = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.stop_event = Event()
        self.sync_event = Event()

    def get(self):
        while True:
            ret, frame = self.stream.read()
            self.sync_event.set()
            yield ret, frame


    def stop(self):
        self.stop_event.set()

'''
class SignPredictor:
    """
    Class that holds and predicts sign recognition.
    """
    def __init__(self, src=0):
        self.stop_event = Event()

        ap = argparse.ArgumentParser()
        ap.add_argument("-w", "--weights", default=None,
                        help="path to the model weights")
        required_ap = ap.add_argument_group('required arguments')
        required_ap.add_argument("-m", "--model",
                                 type=str, default="resnet", required=True,
                                 help="name of pre-trained network to use")
        args = vars(ap.parse_args())
        # ====== Create model for real-time classification ======
        # =======================================================

        if args["model"] not in MODELS:
            raise AssertionError("The --model command line argument should be a key in the `MODELS` dictionary")

        # Create pre-trained model + classification block, with or without pre-trained weights
        self.model = create_model(model=args["model"],
                                model_weights_path=args["weights"])

        # Dictionary to convert numerical classes to alphabet
        self.label_dict = {pos: letter
                      for pos, letter in enumerate(string.ascii_uppercase)}

    def get(self, video):
        
        if not self.ret and not self.stop_event.is_set():
            self.stop()
        else:
            prediction_lock.acquire()
            video_lock.acquire()
            # Crop + process captured frame
            video.frame = square_pad(video.frame)
            video.frame = preprocess_for_vgg(video.frame)

            # Make prediction
            self.prediction = self.model.predict(video.frame,
                                          batch_size=1,
                                          verbose=0)
            video_lock.release()
            prediction_lock.release()


    def stop(self):
        self.stop_event.set()
'''

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

#turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)

#Video Generator Lock
video_lock   = Lock()

#Prediction Generator Thread
prediction_thread = Thread()
prediction_lock   = Lock()

video = VideoController()
#predictor = SignPredictor()

'''
def CyclicSignPredictor():
    """
    Generate a sign prompt as well as a model prediction of the captured sign response,
    and emit to a socketio instance (broadcast)
    """
    global video, predictor

    print("Making random character prompts")
    new_char = True
    attempts = 0

    while not predictor.stop_event.is_set():
        video.sync_event.wait()    
        if new_char == True:
            char = chr(int(random()*26 + 65))
            print(char)
            socketio.emit('newWord', {'char': char}, namespace='/test')
            new_char = False
            attempts = 0

        predictor.get(video)

        # Predict letter
        prediction_lock.acquire()
        top_prd = np.argmax(predictor.prediction)

        # Only display predictions with probabilities greater than 0.5
        if np.max(predictor.prediction) >= 0.50:

            pred_1 = predictor.label_dict[top_prd]
            preds_list = np.argsort(predictor.prediction)[0]
            pred_2 = predictor.label_dict[preds_list[-2]]
            pred_3 = predictor.label_dict[preds_list[-3]]
        prediction_lock.release()

        attempts = attempts + 1

        if pred_1 == char:
            socketio.emit('result', {'char':char,'res': 1},namespace='/test')
            new_char = True
            sleep(2)
        elif attempts > 20:
            socketio.emit('result', {'char':char,'res': 0},namespace='/test')
            sleep(2)
            new_char = True
        else:
            continue
        video.sync_event.clear()
'''                

'''
def PublishToWeb(): 
    global video, predictor
    while True:
        if not video.ret or not predictor.ret:
            break
        else:
            video_lock.acquire()
            prediction_lock.acquire()

            width = int(video.stream.get(3) + 0.5)
            height = int(video.stream.get(4) + 0.5)

            # Annotate image with most probable prediction
            #org=(width // 2 + 230, height // 2 + 75),
            cv2.putText(video.frame, text=predictor.pred_1,
                        org = (0,height), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=6, color=(255, 255, 0),
                        thickness=15, lineType=cv2.LINE_AA)
            # Annotate image with second most probable prediction (displayed on bottom left)
            #org=(width // 2 + width // 5 + 40, (360 + 240)),
            cv2.putText(video.frame, text=predictor.pred_2,
                        org = (200, height), 
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=4, color=(0, 0, 255),
                        thickness=6, lineType=cv2.LINE_AA)
            # Annotate image with third probable prediction (displayed on bottom right)
            #org=(width // 2 + width // 3 + 5, (360 + 240)),
            cv2.putText(video.frame, text=predictor.pred_3,
                        org = (250,height), 
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=4, color=(0, 0, 255),
                        thickness=6, lineType=cv2.LINE_AA)


            video.ret, buffer = cv2.imencode('.jpg', video.frame)
            video.frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + video.frame + b'\r\n')  # concat frame one by one and show result

            video_lock.release()
            prediction_lock.release()
'''

@app.route("/video_feed")
def video_feed():
    global video, ret, frame
    # return the response generated along with the video stream media
    ret, frame = video.get()
    ret, buffer = cv2.imencode('.jpg', frame)
    image = buffer.tobytes()
    feed = (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result
    return Response(feed,
        mimetype = "multipart/x-mixed-replace; boundary=frame")


@app.route('/')
def index():
    #only by sending this page first will the client be connected to the socketio instance
    return render_template('index.html')

@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global video_thread, prediction_thread
    print('Client connected')

'''
    #Start the video thread only if the thread has not been started before.
    if not video_thread.is_alive():
        print("Starting Video Thread")
        video_thread = socketio.start_background_task(video.get)
'''
'''
    #Start the sign prediction generator thread only if the thread has not been started before.
    if not prediction_thread.is_alive():
        print("Starting Predition Thread")
        prediction_thread = socketio.start_background_task(CyclicSignPredictor)
'''
'''
@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    # Release the capture
    video.stop()
    predictor.stop()
    video.stream.release()
    cv2.destroyAllWindows()
    print('Client disconnected')
'''

if __name__ == '__main__':
    socketio.run(app)










#camera.py
# import the necessary packages
import cv2
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from random import random
from time import sleep, time
from threading import Thread, Event, Lock
import cv2
import requests
from processing import square_pad, preprocess_for_vgg
from model import create_model
import argparse
import numpy as np
import string
import queue

global prompt, showingResults, persistance, attempts
prompt = 'a'
showingResults = False
persistance = 0
attempts = 0
waitTime = 2


# Map model names to classes
MODELS = ["resnet", "vgg16", "inception", "xception", "mobilenet"]

class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()

    def getFrame(self, frameQueue):
        while True:
            #extracting frames
            ret, frame = self.video.read()
            frameQueue.put(frame)

class SignPredictor:
    """
    Class that holds and predicts sign recognition.
    """
    def __init__(self, src=0):
        ap = argparse.ArgumentParser()
        ap.add_argument("-w", "--weights", default=None,
                        help="path to the model weights")
        required_ap = ap.add_argument_group('required arguments')
        required_ap.add_argument("-m", "--model",
                                 type=str, default="resnet", required=True,
                                 help="name of pre-trained network to use")
        args = vars(ap.parse_args())
        # ====== Create model for real-time classification ======
        # =======================================================

        if args["model"] not in MODELS:
            raise AssertionError("The --model command line argument should be a key in the `MODELS` dictionary")

        # Create pre-trained model + classification block, with or without pre-trained weights
        self.model = create_model(model=args["model"],
                                model_weights_path=args["weights"])

        # Dictionary to convert numerical classes to alphabet
        self.label_dict = {pos: letter
                      for pos, letter in enumerate(string.ascii_uppercase)}

    def predict(self, frame):
            # Crop + process captured frame
            frame = square_pad(frame)
            frame = preprocess_for_vgg(frame)

            # Make prediction
            prediction = self.model.predict(frame,
                                          batch_size=1,
                                          verbose=0)
            return prediction


def signPredicton(camera, frame, predictor):
    """
    Generate a sign prompt as well as a model prediction of the captured sign response,
    and emit to a socketio instance (broadcast)
    """
    topPrediction = 0
    prediction = predictor.predict(frame)

    # Predict letter
    top_prd = np.argmax(prediction)

    # Only display predictions with probabilities greater than 0.5
    if np.max(prediction) >= 0.50:
        topPrediction = predictor.label_dict[top_prd]

    return topPrediction 


def resetPrompt():
    global prompt

    prompt = chr(int(random()*26 + 65))
    print(prompt)
    socketio.emit('newWord', {'char': prompt}, namespace='/test')
    persistance = 0
    attempts = 0

def promptHandling(topPrediction, showTime):
    global prompt, persistance, attempts, showingResults
    resetResult = False
    
    if not showingResults:
        if topPrediction == prompt:
            if persistance > 5:
                socketio.emit('result', {'char':prompt,'res': 1},namespace='/test')
                showingResults = True
                showTime = time()
            else:
                presistance = persistance + 1
        elif attempts > 50:
            socketio.emit('result', {'char':prompt,'res': 0},namespace='/test')
            showingResults = True
            showTime = time()
        else:
            persistance = 0
        
        attempts = attempts + 1

    else:
        now = time()
        waiting = now - showTime
        if waiting >= waitTime:
            resetResult = True
            showingResults = False

    return resetResult, showTime


def CyclicGeneration(camera, predictor, frameQueue):
    resetResult = True
    showTime = 0
    while True:
        frame = frameQueue.get(True)
        if resetResult == True:
            resetPrompt()
        topPrediction = signPredicton(camera, frame, predictor)
        resetResult, showTime = promptHandling(topPrediction, showTime)
        yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')  # concat frame one by one and show result

def getFrames(camera, predictor, frameQueue):
    #get camera frame
    while True:
        camera.getFrame(frameQueue)
        if not frameQueue.empty():
            frame = frameQueue.get(True)
            ret, buffer = cv2.imencode('.jpg', frame)
            jpeg = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')  # concat frame one by one and show result

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

#turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')

@socketio.on('connect', namespace='/test')
def test_connect():
    print('Client connected')

@app.route('/video_feed')
def video_feed():
    camera = VideoCamera()
    predictor = SignPredictor()
    frameQueue = queue.Queue()
    predictionThread = Thread()
    '''
    #Start the video thread only if the thread has not been started before.
    if not predictionThread.is_alive():
        print("Starting Video Thread")
        predictionThread = socketio.start_background_task(CyclicGeneration, camera, predictor, frameQueue)
        '''
    return Response(getFrames(camera, predictor, frameQueue),
                mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # defining server ip address and port
    app.run()





'''
        preds_list = np.argsort(prediction)[0]
        pred_2 = predictor.label_dict[preds_list[-2]]
        pred_3 = predictor.label_dict[preds_list[-3]]

    height = int(camera.video.get(4) - 20)

    # Annotate image with most probable prediction
    #org=(width // 2 + 230, height // 2 + 75),
    cv2.putText(frame, text=topPrediction,
                org = (50, height), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=6, color=(255, 255, 0),
                thickness=15, lineType=cv2.LINE_AA)
    # Annotate image with second most probable prediction (displayed on bottom left)
    #org=(width // 2 + width // 5 + 40, (360 + 240)),
    cv2.putText(frame, text=pred_2,
                org = (200, height), 
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=4, color=(0, 0, 255),
                thickness=6, lineType=cv2.LINE_AA)
    # Annotate image with third probable prediction (displayed on bottom right)
    #org=(width // 2 + width // 3 + 5, (360 + 240)),
    cv2.putText(frame, text=pred_3,
                org = (250,height), 
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=4, color=(0, 0, 255),
                thickness=6, lineType=cv2.LINE_AA)
    '''