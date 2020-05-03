import pandas as pd
import numpy as np
import tensorflow as tf

def dense_autoencoder():
    input_image = tf.keras.layers.Input(shape=(784,))
    encode = tf.keras.layers.Dense(128,activation="relu")(input_image)
    encode = tf.keras.layers.Dense(128,activation="relu")(encode)
    encode = tf.keras.layers.Dense(256,activation="relu")(encode)
    encode = tf.keras.layers.Dense(64,activation="sigmoid")(encode)
    neck = tf.keras.layers.Dense(32,activation="relu")(encode)
    decode = tf.keras.layers.Dense(64,activation="sigmoid")(neck)
    decode = tf.keras.layers.Dense(128,activation="relu")(decode)
    decode = tf.keras.layers.Dense(128,activation="relu")(decode)
    decode = tf.keras.layers.Dense(784,activation="sigmoid")(decode)
    autoencode = tf.keras.models.Model(input_image,decode)
    encoder = tf.keras.models.Model(input_image,neck)
    return autoencode,encoder
    
    
def dense_encoder():
    input_image = tf.keras.layers.Input(shape=(784,))
    encode = tf.keras.layers.Dense(128,activation="relu")(input_image)
    encode = tf.keras.layers.Dense(128,activation="relu")(encode)
    encode = tf.keras.layers.Dense(256,activation="relu")(encode)
    encode = tf.keras.layers.Dense(64,activation="sigmoid")(encode)
    neck = tf.keras.layers.Dense(32,activation="relu")(encode)
    encoder = tf.keras.models.Model(input_image,neck)
    return encoder

def get_data():
    images_1 = pd.read_csv("./train.csv")
    images_2 = pd.read_csv("./test.csv")
    images_1.pop("label");
    images_1 = images_1.values / 255
    images_2 = images_2.values / 255
    images = np.concatenate([images_1,images_2])
    del images_1,images_2
    return images