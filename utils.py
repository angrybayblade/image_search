import sys
import numpy as np
import pandas as pd
import cv2
import pickle

def dense_autoencoder():
    import tensorflow as tf
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
    import tensorflow as tf
    input_image = tf.keras.layers.Input(shape=(784,))
    encode = tf.keras.layers.Dense(128,activation="relu")(input_image)
    encode = tf.keras.layers.Dense(128,activation="relu")(encode)
    encode = tf.keras.layers.Dense(256,activation="relu")(encode)
    encode = tf.keras.layers.Dense(64,activation="sigmoid")(encode)
    neck = tf.keras.layers.Dense(32,activation="relu")(encode)
    encoder = tf.keras.models.Model(input_image,neck)
    return encoder

def conv_autoencoder():
    import tensorflow as tf
    
    inputs = tf.keras.layers.Input(shape=(28,28,1))
    enc = tf.keras.layers.Conv2D(32,3,activation="relu",padding="same")(inputs)
    enc = tf.keras.layers.MaxPool2D()(enc)
    enc = tf.keras.layers.Conv2D(48,3,activation="linear",padding="same")(enc)
    enc = tf.keras.layers.MaxPool2D()(enc)

    dense = tf.keras.layers.Flatten()(enc)
    dense = tf.keras.layers.Dense(256,activation="relu")(dense)
    encodings = tf.keras.layers.Dense(64,activation="linear")(dense)
    dense = tf.keras.layers.Dense(256,activation="relu")(encodings)
    dense = tf.keras.layers.Dense(1176)(dense)
    dense = tf.keras.layers.Reshape((7,7,24))(dense)

    dec = tf.keras.layers.UpSampling2D()(dense)
    dec = tf.keras.layers.BatchNormalization()(dec)
    dec = tf.keras.layers.Conv2D(48,3,activation="linear",padding="same")(dec)
    dec = tf.keras.layers.UpSampling2D()(dec)
    dec = tf.keras.layers.BatchNormalization()(dec)
    dec = tf.keras.layers.Conv2D(32,3,activation="relu",padding="same")(dec)
    dec = tf.keras.layers.Conv2D(1,4,activation="sigmoid",padding="same")(dec)

    return tf.keras.Model(inputs,dec),tf.keras.Model(inputs,encodings)

def conv_encoder():
    import tensorflow as tf
    
    inputs = tf.keras.layers.Input(shape=(28,28,1))
    enc = tf.keras.layers.Conv2D(32,3,activation="relu",padding="same")(inputs)
    enc = tf.keras.layers.MaxPool2D()(enc)
    enc = tf.keras.layers.Conv2D(48,3,activation="linear",padding="same")(enc)
    enc = tf.keras.layers.MaxPool2D()(enc)

    dense = tf.keras.layers.Flatten()(enc)
    dense = tf.keras.layers.Dense(256,activation="relu")(dense)
    encodings = tf.keras.layers.Dense(64,activation="linear")(dense)
    
    return tf.keras.Model(inputs,encodings)


def get_data():
    images_1 = pd.read_csv("./train.csv")
    images_2 = pd.read_csv("./test.csv")
    images_1.pop("label");
    images_1 = images_1.values / 255
    images_2 = images_2.values / 255
    images = np.concatenate([images_1,images_2])
    del images_1,images_2
    return images


def generate_images():
    from tqdm import tqdm
    images = get_data()
    for i,img in enumerate(tqdm(images)):
        cv2.imwrite(f"./images/img_{i}.jpg",(img.reshape(28,28)*255).astype(np.uint8))

class DenseAutoEncoderSearch(object):
    """
    Dense Auto Encoder Class
    """
    def __init__(self,):
        self.encodings = np.load("./weights/encodings/dense_autoencoder.npy")
        self.kmeans = pickle.load(open(f"./weights/cluster_objects/kmeans_dense_cluster.pickle","rb"))
        self.clusters = self.kmeans.predict(self.encodings)
        _,self.encoder =  dense_autoencoder()
        self.encoder.load_weights("weights/models/dense_autoencoder/dense_encoder")


    def __call__(self,x):
        x = cv2.imread(f"./search/{x}")
        x = x.mean(axis=2)
        x = self.encoder.predict(x.reshape(1,784))
        # print (x)
        x = self.kmeans.predict(x)
        # print (x)
        x = np.where(self.clusters == x)
        return x[0]
    
class ConvolutionalAutoEncoderSearch(object):
    """
    Convolutional Auto Encoder Class
    """
    def __init__(self,):
        self.encodings = np.load("./weights/encodings/conv_autoencoder.npy")
        self.kmeans = pickle.load(open(f"./weights/cluster_objects/kmeans_conv_cluster.pickle","rb"))
        self.clusters = self.kmeans.predict(self.encodings)
        _,self.encoder =  dense_autoencoder()
        self.encoder.load_weights("weights/models/conv_autoencoder/encoder")


    def __call__(self,x):
        x = cv2.imread(f"./search/{x}")
        x = x.mean(axis=2)
        x = self.encoder.predict(x.reshape(1,28,28,1))
        # print (x)
        x = self.kmeans.predict(x)
        # print (x)
        x = np.where(self.clusters == x)
        return x[0]
    
if __name__ == "__main__":
    if "--generate_images" in sys.argv:
        generate_images()
    
    dae = DenseAutoEncoderSearch()
    print (dae("img_10012.jpg"))