# Import packets and libraries
# Import packets and libraries
import os
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU, Conv2D, UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2,ResNet50, VGG19

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K

import argparse
from sklearn.metrics import pairwise_distances_argmin_min
import tensorflow.keras.layers as L
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D
from keras.losses import binary_crossentropy, BinaryCrossentropy
import pandas
from tensorflow.keras.callbacks import ModelCheckpoint
tf.compat.v1.enable_eager_execution()
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # run the code on a specified GPU
# Check the list of available GPUs.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available")
else:
    print("GPU is not available")



class Augment(tf.keras.layers.Layer): # A data augmentation layer of a tensor

    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        # Create a RandomFlip layer to augment both images and masks
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)
        # The images are augmented by horizontal and vertical flipping, and rotated within the range of -0.2 to 0.2 radians.
        self.augumet = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal_and_vertical"),tf.keras.layers.RandomRotation(0.2),])
    def call(self, inputs, labels):
        inputs = self.augumet(inputs)
        labels = self.augumet(labels)
        return inputs, labels

class SiLUActivationLayer(L.Layer):
    def call(self, inputs):
        return tf.nn.silu(inputs)

def conv_block(x, num_filters):
    x = L.Conv2D(num_filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x) # Helps stabilize and accelerate learning by normalizing the output values after the convolutional layer.
    x = L.Activation("relu")(x) # Apply the ReLU (Rectified Linear Unit) activation function after batch normalization.
    x = L.Conv2D(num_filters, 3, padding="same")(x) # Learn more complex features after the first layer.
    x = L.BatchNormalization()(x) # Maintain stability and accelerate the model.
    x = L.Activation("relu")(x) # Activation increases the non-linearity of the model.
    return x #The output is a tensor X containing feature information extracted from the input.

def encoder_block(x, num_filters): # Construct encoder blocks.
    x = conv_block(x, num_filters) # Create a convolutional block to learn features from the input data.
    p = L.MaxPool2D((2, 2))(x) # It is applied to reduce the size of the feature map.
    return x, p # The output of the convolutional block and the max-pooling layer.

def attention_gate(g, s, num_filters):  # g is the gating signal, and s is the feature from the encoder.
    # Attention weighting algorithm: Wg and Ws

    # Apply a convolutional layer with a 1×1 kernel on the encoder feature s to generate attention weights.
    Wg = L.Conv2D(num_filters, 1, padding="same")(g) # The kernel size is 1×1 to generate a single attention weight for each pixel.
    Wg = L.BatchNormalization()(Wg) # To stabilize and accelerate the learning of the network.
    # Apply a convolutional layer with a 1×1 kernel to the encoder feature s to generate attention weights.
    Ws = L.Conv2D(num_filters, 1, padding="same")(s)
    Ws = L.BatchNormalization()(Ws) # to stabilize and accelerate the learning process of the network
    # Kết hợp trọng số
    out = L.Activation("relu")(Wg + Ws) 
    out = L.Conv2D(num_filters, 1, padding="same")(out) 
    out = L.Activation("sigmoid")(out)
    
    return out * s
    
def decoder_block(x, s, num_filters): 
    x = L.UpSampling2D(interpolation="bilinear")(x) 
    s = attention_gate(x, s, num_filters) 
    x = L.Concatenate()([x, s]) 
    x = conv_block(x, num_filters) 
    return x 

# Spatial Pyramid Pooling (SPP): Addresses the issue of fixed input size and enhances the model’s spatial representation capability.
def SPPF_module(x, k = 5):
    B, H, W, C = x.shape # Batch - Height - Width - Channel
    cv1 = Conv2D(C//2, (1,1), strides=(1,1), padding = 'valid')(x) 
    cv1 = BatchNormalization()(cv1) 
    cv1 = SiLUActivationLayer()(cv1) 
    
    mp1 = MaxPooling2D(pool_size = (k, k), strides = (1, 1), padding = 'same')(cv1) 
    mp2 = MaxPooling2D(pool_size = (k, k), strides = (1, 1), padding = 'same')(mp1)
    mp3 = MaxPooling2D(pool_size = (k, k), strides = (1, 1), padding = 'same')(mp2)

    out = Concatenate(axis=-1)([cv1, mp1, mp2, mp3])
    
    out = Conv2D(C, (1,1), strides=(1,1), padding = 'valid')(out)

    out = BatchNormalization()(out) 
    out = SiLUActivationLayer()(out) 
    return out # The function returns a feature map processed by the SPP module, containing spatial information.

def build_CSA_attention_unet(input_shape):
    """ Input """
    inputs = Input(input_shape) # (Height, Width, Channel)

    """ Pre-trained VGG19 Model """
    CSA = VGG19(include_top = False, weights = "imagenet", input_tensor = inputs)

    """ Encoder """
    
    s1 = CSA.get_layer("block1_conv2").output
    # s1 = SPPF_module(s1, k = 5)
    s2 = CSA.get_layer("block2_conv2").output
    # s2 = SPPF_module(s2, k = 5)
    s3 = CSA.get_layer("block3_conv4").output
    # s3 = SPPF_module(s3, k = 5)
    s4 = CSA.get_layer("block4_conv4").output
    # s4 = SPPF_module(s4, k = 5)
    b1 = CSA.get_layer("block5_conv4").output
    b1 = SPPF_module(b1, k = 5)

  
    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Output """
    
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs) 
    return model






