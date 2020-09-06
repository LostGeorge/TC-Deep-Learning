import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import cv2
import os, sys

INPUT_DIM = 128
N_SENSORS = 2

# TODO: Image pipeline - create necessary tensors
#       Data pipeline - one hot encode sensor number, standardize feats to [0, 1], create tensors
#       Compile Model - figure out optimizer, evaluation metric

path = sys.argv[1]
imgs = [np.load(path + '/' + d) for d in os.listdir(path)]

imgs = [cv2.resize(i, (INPUT_DIM, INPUT_DIM), interpolation=cv2.INTER_NEAREST) for i in imgs]

img_inputs = keras.Input(shape=(INPUT_DIM, INPUT_DIM, 1))
x = layers.Conv2D(8, 3, padding='valid', activation='relu')(img_inputs)
x = layers.MaxPool2D(2)(x)
x = layers.Conv2D(16, 5, padding='valid', activation='relu')(x)
x = layers.MaxPool2D(3)(x)
x = layers.Conv2D(32, 5, padding='valid', activation='relu')(x)
x = layers.MaxPool2D(3)(x)
x = layers.Flatten()(x)

data_inputs = keras.Input(shape=(5 + N_SENSORS,))
y = layers.Dense(32, activation='relu')(data_inputs)

combined = layers.concatenate([x, y])
combined = layers.Dense(256, activation='relu')(combined)
combined = layers.Dense(64, activation='relu')(combined)
combined = layers.Dense(16, activation='relu')(combined)
output = layers.Dense(7, activation='sigmoid')(combined)

model = keras.Model(inputs=[img_inputs, data_inputs], outputs=output)

print(model.summary())
