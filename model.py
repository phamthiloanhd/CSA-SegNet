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
# Kiểm tra danh sách các GPU có sẵn
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available")
else:
    print("GPU is not available")



class Augment(tf.keras.layers.Layer): # Một tầng tăng cường dữ liệu của Tensor

    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        # Tạo một lớp RandomFlip để tăng cường dữ liệu của images và mask
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)
        # Ảnh được tăng cường bằng lật ngang - dọc và quay -0.2->0.2
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
    x = L.BatchNormalization()(x) # Giúp ổn định, tăng tốc độ học bằng cách chuẩn hóa giá trị đầu ra sau lớp Convo
    x = L.Activation("relu")(x) # Áp dụng hàm kích hoạt ReLU (Rectified Linear Unit) sau batch normalization

    x = L.Conv2D(num_filters, 3, padding="same")(x) # Học các đặc trưng phức tạp hơn sau lớp thứ nhất
    x = L.BatchNormalization()(x) # Duy trì sự ổn định và tăng tốc mô hình
    x = L.Activation("relu")(x) # Kích hoạt tăng tính phi tuyến của mô hình
    return x # Đầu ra là tensor X chứa các thông tin dữ liệu đặc trưng (feature) từ đầu vào

def encoder_block(x, num_filters): # Xây dựng các khối mã hóa (encoder blocks)
    x = conv_block(x, num_filters) # tạo ra một khối convolution để học các đặc trưng của dữ liệu đầu vào.
    p = L.MaxPool2D((2, 2))(x) # Được áp dụng để giảm kích thước của feature map
    return x, p # Đầu ra của khối convolution và lớp max-pooling

def attention_gate(g, s, num_filters):  # g là gating signal - s là phần từ encoder
    # Thuật toán trọng số chú ý: Wg and Ws

    # Thực hiện lớp convo trên phần tử từ encoder s với kernel size 1x1 để tạo trọng số
    Wg = L.Conv2D(num_filters, 1, padding="same")(g) # Kernel size là 1x1 để chỉ tạo ra một trọng số chú ý cho mỗi pixel
    Wg = L.BatchNormalization()(Wg) # Để ổn định hóa và tăng tốc độ học của mạn

    # Thực hiện lớp convo trên phần tử từ encoder s với kernel size 1x1 để tạo trọng số
    Ws = L.Conv2D(num_filters, 1, padding="same")(s)
    Ws = L.BatchNormalization()(Ws) # để ổn định hóa và tăng tốc độ học của mạng

    # Kết hợp trọng số
    out = L.Activation("relu")(Wg + Ws) # Giúp tăng tính phi tuyến tính và kết hợp thông tin từ cả hai đầu vào
    out = L.Conv2D(num_filters, 1, padding="same")(out) # Thực hiện một lớp convolution với kernel size 1x1
    out = L.Activation("sigmoid")(out)
    # Áp dụng hàm kích hoạt sigmoid để đảm bảo rằng giá trị đầu ra nằm trong khoảng từ 0 đến 1, thể hiện trọng số chú ý cho mỗi pixel

    return out * s
    # Tích của trọng số chú ý và phần tử từ encoder s - Điều này đảm bảo rằng thông tin từ s được điều chỉnh theo trọng số chú ý trước khi được truyền vào các phần tử khác của mạng.

# Tăng kích thước của feature maps và học các đặc trưng chi tiết để phục hồi thông tin từ encoder
def decoder_block(x, s, num_filters): # x là đầu ra của decoder, s là từ encoder
    # interpolation="bilinear" sử dụng để up-sampling bằng cách sử dụng phương pháp tuyến tính nội suy hai chiều, giữ nguyên tỷ lệ giữa các giá trị
    x = L.UpSampling2D(interpolation="bilinear")(x) # Tăng kích thước của feature maps
    s = attention_gate(x, s, num_filters) # Mục tiêu là điều chỉnh thông tin từ encoder trước khi nó được kết hợp với thông tin từ decoder.
    # Điều này giúp kết hợp thông tin chi tiết từ encoder với thông tin từ decoder để tạo ra một feature map phong phú hơn và chứa nhiều thông tin hơn
    x = L.Concatenate()([x, s]) # Kết hợp feature maps từ decoder và encoder bằng cách concatenate chúng lại với nhau
    x = conv_block(x, num_filters) # Thực hiện một convolution block trên feature map đã được concatenate để học các đặc trưng chi tiết
    return x # Hàm trả về feature map đã được xử lý từ khối giải mã, chứa các đặc trưng được học để phục hồi thông tin từ encoder

# Spatial Pyramid Pooling (SPP): Giải quyết vấn đề về kích thước không đổi của đầu vào và tăng cường khả năng biểu diễn không gian model
def SPPF_module(x, k = 5):
    B, H, W, C = x.shape # Batch - Height - Width - Channel

    # Thực hiện một lớp convolution với kernel size là (1,1) để giảm số lượng channels xuống cột 1/2
    cv1 = Conv2D(C//2, (1,1), strides=(1,1), padding = 'valid')(x) # không có padding được thêm vào xung quanh đầu vào
    cv1 = BatchNormalization()(cv1) # Để ổn định và tăng tốc độ của mạng
    #cv1 = tf.nn.silu(cv1) # Hàm kích hoạt SiLU (Sigmoid Linear Unit) cho đầu ra của convolutional layer
    cv1 = SiLUActivationLayer()(cv1) # Hàm kích hoạt SiLU (Sigmoid Linear Unit) cho đầu ra của convolutional layer

    # Thực hiện max pooling với kernel size là (k, k) để tạo feature map với kích thước giảm mà vẫn giữ thông tin không gian
    mp1 = MaxPooling2D(pool_size = (k, k), strides = (1, 1), padding = 'same')(cv1) # padding được thêm vào xung quanh đầu vào sao cho output có kích thước bằng với input
    mp2 = MaxPooling2D(pool_size = (k, k), strides = (1, 1), padding = 'same')(mp1)
    mp3 = MaxPooling2D(pool_size = (k, k), strides = (1, 1), padding = 'same')(mp2)

    # Kết hợp các feature maps từ convolutional layer và các max pooling layers
    out = Concatenate(axis=-1)([cv1, mp1, mp2, mp3])

    # Thực hiện một lớp convolution cuối cùng để kết hợp thông tin từ các feature maps (c, kernal, strides, padding)
    out = Conv2D(C, (1,1), strides=(1,1), padding = 'valid')(out)

    out = BatchNormalization()(out) # Ổn định và tăng tốc độ mạng
    #out = tf.nn.silu(out) # Áp dụng hàm kích hoạt SiLU cho đầu ra
    out = SiLUActivationLayer()(out) # Áp dụng hàm kích hoạt SiLU cho đầu ra
    return out # hàm trả về feature map đã được xử lý từ mô-đun SPP, chứa thông tin không gian

def build_vgg19_attention_unet(input_shape):
    """ Input """
    inputs = Input(input_shape) # Kích thước đầu vào (Height, Width, Channel)

    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top = False, weights = "imagenet", input_tensor = inputs)

    """ Encoder """
    # Lấy ra các tensor đầu ra từ các lớp convolution trong mô hình VGG16 (s1, s2, s3, s4) để sử dụng làm feature maps của bộ mã hóa
    # Mỗi đầu ra của lớp sẽ có trọng số khác nhau
    s1 = vgg19.get_layer("block1_conv2").output
    # s1 = SPPF_module(s1, k = 5)
    s2 = vgg19.get_layer("block2_conv2").output
    # s2 = SPPF_module(s2, k = 5)
    s3 = vgg19.get_layer("block3_conv3").output
    # s3 = SPPF_module(s3, k = 5)
    s4 = vgg19.get_layer("block4_conv3").output
    # s4 = SPPF_module(s4, k = 5)
    b1 = vgg19.get_layer("block5_conv3").output
    b1 = SPPF_module(b1, k = 5)

    """ Bridge """
    # b1 = vgg19.get_layer("block5_conv4").output

    """ Decoder """
    # Nhận feature maps từ bộ mã hóa và tiến hành giải mã để tạo ra các feature maps có kích thước tương ứng giảm dần
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Output """
    # Lớp convolution sử dụng để tạo ra đầu ra của mô hình, có kích thước tương tự như input và được kích hoạt bằng hàm sigmoid để tạo ra một dự đoán nhị phân cho mỗi pixel
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs) # Model từ các tensor đầu vào và đầu ra, tạo thành một mô hình hoàn chỉnh.
    return model






