import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import cv2
import os, sys

INPUT_DIM = 601
RSZ_DIM = 128
N_IMG_TYPES = 1
N_SENSORS = 1

# TODO: [Done] Image pipeline - create necessary tensors
#       Data pipeline - one hot encode sensor number, standardize feats to [0, 1], create tensors
#       Compile Model - figure out optimizer, evaluation metric
#       Full Data Integration - modify to pull from remote s3 data and figure out

img_path = sys.argv[1]
image_types = os.listdir(img_path)
N_IMG_TYPES = len(image_types)
imgs = []
for i_path in os.listdir('/'.join([img_path, image_types[0]])):
    img_stack = np.zeros((INPUT_DIM, INPUT_DIM, N_IMG_TYPES), dtype=np.uint8)
    for j, typ in enumerate(image_types):
        img_stack[:, :, j] = np.load('/'.join([img_path, typ, i_path]))
    imgs.append(img_stack)

imgs = [cv2.resize(img, (RSZ_DIM, RSZ_DIM), interpolation=cv2.INTER_NEAREST) for img in imgs]
imgs = np.array(imgs)
imgs = tf.convert_to_tensor(imgs, dtype=np.float32)

img_inputs = keras.Input(shape=(RSZ_DIM, RSZ_DIM, N_IMG_TYPES))
x = layers.Conv2D(8, 3, padding='valid', activation='relu')(img_inputs)
x = layers.MaxPool2D(2)(x)
x = layers.Conv2D(16, 5, padding='valid', activation='relu')(x)
x = layers.MaxPool2D(3)(x)
x = layers.Conv2D(32, 5, padding='valid', activation='relu')(x)
x = layers.MaxPool2D(3)(x)
x = layers.Flatten()(x)

# ==================================================================

data_inputs = keras.Input(shape=(5 + N_SENSORS,))
y = layers.Dense(32, activation='relu')(data_inputs)

combined = layers.concatenate([x, y])
combined = layers.Dense(256, activation='relu')(combined)
combined = layers.Dense(64, activation='relu')(combined)
combined = layers.Dense(16, activation='relu')(combined)
output = layers.Dense(7, activation='sigmoid')(combined)

model = keras.Model(inputs=[img_inputs, data_inputs], outputs=output)

print(model.summary())
