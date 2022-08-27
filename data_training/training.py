import numpy as np
import os
# 0-size
import sys 

from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical

def training():
	data_size = 100
	init = False

	l = []
	y = []
	d = {}
	c = 0
	for i in os.listdir():
		if i.split('.')[-1] == "npy":
			l.append(i.split(".")[0])

			d[i.split(".")[0]] = c 
			c = c+1

			if not(init):
				a = np.load(i)
				y = np.array([str(i.split(".")[0])]*data_size)
				init = True
			else:
				a = np.concatenate((a, np.load(i)))
				y = np.concatenate((y, np.array([str(i.split(".")[0])]*data_size)))

			print(i.split(".")[0] + " --> " , a.shape, y.shape)

	print("Dictionary is : ", d)

	for m,n in enumerate(y):
		y[m] = d[n]
	y = to_categorical(y)

	a = np.array(a)
	y = np.array(y)

	print("="*100)
	print("final data : ", a.shape, y.shape)
	print(a.dtype, y.dtype)
	print("="*200)

	i = Input(shape=(1020))

	x = Dense(500, activation="tanh")(i)

	op = Dense(len(l), activation="softmax")(x)

	model = Model(i, op)

	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
	model.summary()
	model.fit(a, y, epochs=100)

	model.save("model.h5")
	np.save("labels.npy", np.array(l))