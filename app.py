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

    def get_frame(self):
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


def CyclicSignPredictor(camera, frame, predictor):
    """
    Generate a sign prompt as well as a model prediction of the captured sign response,
    and emit to a socketio instance (broadcast)
    """
    topPrediction = '*'
    pred_2 = '*'
    pred_3 = '*'
    prediction = predictor.predict(frame)

    # Predict letter
    top_prd = np.argmax(prediction)

    # Only display predictions with probabilities greater than 0.5
    if np.max(prediction) >= 0.50:

        topPrediction = predictor.label_dict[top_prd]
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

    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return frame, topPrediction 


def resetPrompt():
    global prompt, attempts, persistance

    prompt = chr(int(random()*26 + 65))
    print(prompt)
    socketio.emit('newWord', {'char': prompt}, namespace='/test')
    persistance = 0
    attempts = 0

def PromptHandling(topPrediction, showTime):
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
        elif attempts > 25:
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


def gen(camera, predictor):
    #get camera frame
    resetResult = True
    showTime = 0
    while True:
        frame = camera.get_frame()
        if resetResult == True:
            resetPrompt()
        jpeg, topPrediction = CyclicSignPredictor(camera, frame, predictor)
        resetResult, showTime = PromptHandling(topPrediction, showTime)
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
    return Response(gen(camera, predictor),
                mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # defining server ip address and port
    app.run()