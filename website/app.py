import secrets
from flask import Flask,render_template,redirect,request,Response,session
import pyrebase
import cv2
import requests
import json
import cv2
import math
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from keras.models import load_model 
import time
from playsound import playsound
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU


app=Flask(__name__)

def calculate_angle(a,b,c):#shoulder, elbow, wrist
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle    
    return angle

def calculate_angle(a,b,c):#shoulder, elbow, wrist
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle    
    return angle

def calculate_distance(a,b):
    a = np.array(a)
    b = np.array(b)
    print(a)
    print(b)
    
    #distance = ((((b[0] - a[0])**(2)) - ((b[1] - a[1])**(2)))**(0.5))
    distance = math.hypot(b[0] - a[0], b[1] - a[1])



def generate_frames():
    time.sleep(2)
    cap = cv2.VideoCapture('./static/assets/Countdown5.mp4')
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, res1 = cap.read()
        if ret == True:
            _,buffer=cv2.imencode(".jpg",res1)
            res1=buffer.tobytes()
            yield(b' --frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+res1+b'\r\n')
        else: 
            break
    cap.release()
    cv2.destroyAllWindows()    
    camera=cv2.VideoCapture(1)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
            ## read the camera frame
            ret,frame=camera.read()
            image1 = frame
            if not ret:
                break
            else:
                image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting BGR to RGB so that it becomes easier for library to read the image1
                image1.flags.writeable = False #this step is done to save some memoery
                # Make detection
                results = pose.process(image1) #We are using the pose estimation model 
                # Recolor back to BGR
                image1.flags.writeable = True
                image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark                    
                except:
                    pass
                mp_drawing.draw_landmarks(image1, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )  
                _,buffer=cv2.imencode(".jpg",image1)
                image1=buffer.tobytes()
            yield(b' --frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+image1+b'\r\n')
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows()
    
def detection():
    time.sleep(2)
    cap = cv2.VideoCapture('./static/assets/Countdown5.mp4')
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, res1 = cap.read()
        if ret == True:
            _,buffer=cv2.imencode(".jpg",res1)
            res1=buffer.tobytes()
            yield(b' --frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+res1+b'\r\n')
        else: 
            break
    cap.release()
    cv2.destroyAllWindows()  
    labels = np.load('labels.npy')
    model = load_model('model.h5')
    cap = cv2.VideoCapture(1)
    holis = mp.solutions.holistic
    hands = mp.solutions.hands
    drawing = mp.solutions.drawing_utils 
    holisO = holis.Holistic(static_image_mode=False)
    while True:
        x = []
        stime = time.time()
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        res = holisO.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        drawing.draw_landmarks(frame, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frame, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        # drawing.draw_landmarks(frame, res.face_landmarks, holis.FACEMESH_CONTOURS)
        if res.face_landmarks:
            if not(res.left_hand_landmarks):
                for i in range(42):
                    x.append(0.0)
            else:
                lox, loy = res.left_hand_landmarks.landmark[8].x, res.left_hand_landmarks.landmark[8].y
                for i in res.left_hand_landmarks.landmark:
                    x.append(i.x - lox)
                    x.append(i.y - loy)
            if not(res.right_hand_landmarks):
                
                for i in range(42):
                    x.append(0.0)
            else:
                rox, roy = res.right_hand_landmarks.landmark[8].x, res.right_hand_landmarks.landmark[8].y
                for i in res.right_hand_landmarks.landmark:
                    x.append(i.x - rox)
                    x.append(i.y - roy)
            for i in res.face_landmarks.landmark:
                x.append(i.x - res.face_landmarks.landmark[1].x)
                x.append(i.y - res.face_landmarks.landmark[1].y)
            pred = model.predict(np.array([x]))
            cv2.putText(frame, labels[np.argmax(pred)], (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,225,0), 7)
            
        etime = time.time()
        cv2.putText(frame, f"{int(1/(etime-stime))}", (50,340), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        _,buffer=cv2.imencode(".jpg",frame)
        frame=buffer.tobytes()
        yield(b' --frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
        
        


@app.route('/')
def index():
    return render_template("base.html")

@app.route('/video')
def video():
    return Response(detection(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)