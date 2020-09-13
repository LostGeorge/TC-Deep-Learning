import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
import cv2
import os, sys

INPUT_DIM = 601
RSZ_DIM = 128
N_IMG_TYPES = 1
N_SENSORS = 1

# TODO: [Done] Image pipeline - create necessary tensors
#       [Done] Data pipeline - one hot encode sensor number, standardize feats to [0, 1], create tensors
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

data_path = sys.argv[2]
dtypes = {
    'latitude': np.float64,
    'longitude': np.float64,
    'sensor_num': np.int32,
    'sat_lat': np.float64,
    'sat_lon': np.float64,
    'wind_speed': np.float64
}
wind_df = pd.read_csv(data_path + '/wind_data.csv', header=0, index_col=0, dtype=dtypes)
#print(wind_df.head())
label_enc = LabelEncoder()
sensor_col = wind_df['sensor_num'].to_numpy().reshape((len(wind_df.index), 1))
label_enc.fit(sensor_col)
N_SENSORS = len(label_enc.classes_)
wind_df['sensor_num'] = np.ravel(label_enc.transform(sensor_col))
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
sensor_col = wind_df['sensor_num'].to_numpy().reshape((len(wind_df.index), 1))
ohe.fit(sensor_col)
sensor_ohe_arr = ohe.transform(sensor_col)

wind_df = wind_df.drop(columns=['sensor_num'])
s_scaler = StandardScaler(copy=False)
s_scaler.fit(wind_df.to_numpy()[:, :4])
feat_df = pd.DataFrame(s_scaler.transform(wind_df.to_numpy()[:, :4]), columns=wind_df.columns[:-1])

sensor_labels = ['sensor_' + str(i) for i in range(N_SENSORS)]
feat_df = pd.concat([feat_df, pd.DataFrame(sensor_ohe_arr, columns=sensor_labels)], axis=1)
#print(feat_df.head())

labels = np.zeros(len(feat_df.index))
thresholds = [34, 64, 83, 96, 113, 137]
labels[wind_df['wind_speed'] < 34] = 0 # Depressions
for i in range(1, 6): # Ts through Cat4
    labels[np.logical_and(wind_df['wind_speed'] >= thresholds[i-1],
        wind_df['wind_speed'] < thresholds[i])] = i
labels[wind_df['wind_speed'] >= 137] = 6 # Cat5
#print(labels[:5])

data_inputs = keras.Input(shape=(5 + N_SENSORS,))
y = layers.Dense(32, activation='relu')(data_inputs)

combined = layers.concatenate([x, y])
combined = layers.Dense(256, activation='relu')(combined)
combined = layers.Dense(64, activation='relu')(combined)
combined = layers.Dense(16, activation='relu')(combined)
output = layers.Dense(7, activation='sigmoid')(combined)

model = keras.Model(inputs=[img_inputs, data_inputs], outputs=output)

#print(model.summary())
