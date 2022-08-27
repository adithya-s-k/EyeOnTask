import mediapipe as mp 
import numpy as np 
import cv2

def data_collection():
	cap = cv2.VideoCapture(0)

	data_name = input("Enter data name ! ")

	holistic = mp.solutions.holistic
	hands = mp.solutions.hands
	holis = holistic.Holistic()
	drawing = mp.solutions.drawing_utils


	y = []
	c = 0
	while True:
		x = []
		_, frm = cap.read()
		frm = cv2.flip(frm, 1)

		res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))  

		if res.face_landmarks:
			if not(res.left_hand_landmarks):
				print("adding zeroes")
				for i in range(42):
					x.append(0.0)
			else:
				for i in res.left_hand_landmarks.landmark:
					x.append(i.x - res.left_hand_landmarks.landmark[8].x)
					x.append(i.y - res.left_hand_landmarks.landmark[8].y)
			if not(res.right_hand_landmarks):
				
				for i in range(42):
					x.append(0.0)
			else:
				for i in res.right_hand_landmarks.landmark:
					x.append(i.x - res.right_hand_landmarks.landmark[8].x)
					x.append(i.y - res.right_hand_landmarks.landmark[8].y)

			for i in res.face_landmarks.landmark:
				x.append((i.x - res.face_landmarks.landmark[1].x))
				x.append((i.y - res.face_landmarks.landmark[1].y))

			c = c+1
			y.append(x)


		drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
		drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
		drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
		cv2.putText(frm, str(c), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
		cv2.imshow("window", frm)
		if cv2.waitKey(1) == 27 or c>99:
			cap.release()
			cv2.destroyAllWindows()
			break

	y = np.array(y)
	print("="*50)
	print(y.shape)
	print(y[0])
	np.save(data_name+".npy", y)