import math
import cv2 
import mediapipe as mp 
import numpy as np
from keras.models import load_model
import time 
from keras.layers import LeakyReLU
from matplotlib import pyplot as plt

timing = []
list_State = []
state = "screen"
prevState = "not screen"
begin = 0
end = 0
stateDisp = False
dict_stats={}
very_begin = 0

def start(): 
    global begin
    begin=time.time()

def ending():
    global end
    end=time.time()
    global elapsed
    elapsed=end-begin
    elapsed=int(elapsed)
    timing.append(int(elapsed))

def calculate_angle(a,b,c):#shoulder, elbow, wrist # type: ignore
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
    return distance

def inference():
	very_begin = time.time()
	labels = np.load('labels.npy')
	model = load_model('model.h5')
	cap = cv2.VideoCapture(0)
	holis = mp.solutions.holistic  # type: ignore
	hands = mp.solutions.hands # type: ignore
	drawing = mp.solutions.drawing_utils # type: ignore
	holisO = holis.Holistic(static_image_mode=False)
	while True:
		x = []
		stime = time.time()
		ret, frame = cap.read()
		if ret == True:
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
				pred = model.predict(np.array([x])) # type: ignore
				cv2.putText(frame, labels[np.argmax(pred)], (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,225,0), 7)

				state = str(labels[np.argmax(pred)])
				list_State.append(state)
				if(state == "screen" and prevState == "not away"):
					start()
				elif(state == "not away" and prevState == "screen"):
					ending()
		    
			etime = time.time()
			cv2.putText(frame, f"{int(1/(etime-stime))}", (50,340), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
			
			try:
				img = cv2.imread(f"emojis/{labels[np.argmax(pred)]}.png") # type: ignore
				img = cv2.resize(img, (100,100))
				frame[100:200, :100] = cv2.addWeighted(frame[100:200, :100], 0, img, 1, 0)
			except:
				pass
			cv2.imshow("window", frame)

			if cv2.waitKey(1) == 27:
				cap.release()
				cv2.destroyAllWindows()
				break

def get_prediction():
    global very_begin
    if len(timing):
        timing.pop(0)
    print(timing)
    time_look_away = sum(timing)
    dict_stats['time_look_away'] = time_look_away
    print("Amount of time looking away",time_look_away)
    very_end_time=time.time()
    overall_timing = int(very_end_time-very_begin)
    
    dict_stats['overall_timing'] = overall_timing
    
    print("over all session timing",overall_timing)
    
    productivity = ((overall_timing-time_look_away)/overall_timing)*100
    print("time prductivity rating",productivity)

    dict_stats['productivity'] = int(productivity)
    
    count_look_away = list_State.count("not screen")
    
    true_productivity = ((count_look_away/len(list_State))*100)
    print("true productivity",true_productivity)
    
    dict_stats['true productivity'] = true_productivity
    
    best_prediction = (true_productivity+productivity)/2
    # print("final productivity prediction", best_prediction)
    
    dict_stats['best Prediction'] = int(best_prediction)
    
    plt.plot([x for x in range(0,len(list_State))],list_State)
    plt.xlabel("time")
    plt.ylabel("State")
    plt.show()
    
    print(dict_stats)