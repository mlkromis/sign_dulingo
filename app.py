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

    def getFrame(self):
            #extracting frames
            ret, frame = self.video.read()
            return frame

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
    topPrediction = ' '
    prediction = predictor.predict(frame)

    # Predict letter
    top_prd = np.argmax(prediction)

    # Only display predictions with probabilities greater than 0.5
    if np.max(prediction) >= 0.50:
        topPrediction = predictor.label_dict[top_prd]

    return topPrediction 


def resetPrompt():
    global prompt, attempts, persistance

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


def CyclicGeneration(camera, predictor, frameQueue, predictionQueue):
    resetResult = True
    showTime = 0
    while True:
        if not frameQueue.empty():
            frame = frameQueue.get(True)
            if resetResult == True:
                resetPrompt()
            topPrediction = signPredicton(camera, frame, predictor)
            predictionQueue.put(topPrediction)
            resetResult, showTime = promptHandling(topPrediction, showTime)

def getFrames(camera, predictor, frameQueue, predictionQueue):
    #get camera frame
    topPrediction = ' '
    while True:
        frame = camera.getFrame()
        frameQueue.put(frame)
        if not predictionQueue.empty():
            topPrediction = predictionQueue.get(True)

        # Annotate image with most probable prediction
        cv2.putText(frame, text=topPrediction,
                    org = (50,200), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=6, color=(255, 255, 0),
                    thickness=15, lineType=cv2.LINE_AA)
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
    predictionQueue = queue.Queue()
    predictionThread = Thread()

    #Start the video thread only if the thread has not been started before.
    if not predictionThread.is_alive():
        print("Starting Video Thread")
        predictionThread = socketio.start_background_task(CyclicGeneration, camera, predictor, frameQueue, predictionQueue)

    return Response(getFrames(camera, predictor, frameQueue, predictionQueue),
                mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # defining server ip address and port
    app.run()
