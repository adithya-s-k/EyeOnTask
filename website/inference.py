import cv2 
import mediapipe as mp 
import numpy as np
from tensorflow.keras.models import load_model
import time 
from tensorflow.keras.layers import LeakyReLU

def inference():
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

		_, frame = cap.read()
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
		
		try:
			img = cv2.imread(f"emojis/{labels[np.argmax(pred)]}.png")
			img = cv2.resize(img, (100,100))
			frame[100:200, :100] = cv2.addWeighted(frame[100:200, :100], 0, img, 1, 0)
		except:
			print("no")
		cv2.imshow("window", frame)

		if cv2.waitKey(1) == 27:
			cap.release()
			cv2.destroyAllWindows()
			break