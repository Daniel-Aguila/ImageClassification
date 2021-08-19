import matplotlib.pyplot as plt
import cv2
import time
from concurrent import futures
import pathlib
import numpy as np

from keras import models, layers, losses
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

def wbce( y_true, y_pred, weight1=500, weight0=1):
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
    return K.mean( logloss, axis=-1)

def model():
    inputs = layers.Input(shape=(2048,1362,1))

    c = layers.Conv2D(64, (3,3), padding="same", strides=(1,1), activation="relu")(c)
    c = layers.Conv2D(64, (3,3), padding="same", strides=(1,1), activation="relu")(inputs)
    c = layers.Conv2D(64, (3,3), padding="same", strides=(1,1), activation="relu")(c)
    c = layers.Conv2D(64, (3,3), padding="same", strides=(1,1), activation="relu")(c)
    c = layers.Conv2D(64, (3,3), padding="same", strides=(1,1), activation="relu")(c)
    c = layers.Dense(100,activation="relu")(c)
    c = layers.Dense(100,activation="relu")(c)
    c = layers.Dense(100,activation="relu")(c)
    c = layers.Dense(10,activation="relu")(c)    
    outputs = layers.Dense(1,activation="sigmoid")(c)
    model = models.Model(inputs, outputs)

    model.compile(loss=wbce,optimizer='adam', metrics=['accuracy'])
    model.summary()
    plot_model(model,to_file='generic_dense_net.png')

def preprocessing(img):
    
    #already converted to grayscale
    #normalize to [0,1] entries
    img = cv2.imread(img)
    img = img.astype("float32")/255
def main():
    start = time.perf_counter()
    dog_image_path = "images\dogs"
    images = pathlib.Path(dog_image_path).iterdir()

#Multiprocessing
    with futures.ProcessPoolExecutor() as executor:
        results = executor.map(preprocessing, images)
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds(s)')
#2048x1362
if __name__ == "__name__":
    main()