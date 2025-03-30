# Import packets and libraries
# Import packets and libraries
import os
import os
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # run the code on a specified GPU

import tensorflow as tf

# Kiểm tra danh sách các GPU có sẵn
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available")
else:
    print("GPU is not available")



import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2,ResNet50

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K
tf.compat.v1.enable_eager_execution()
# Disables eager execution.
# tf.enable_eager_execution()
import argparse

# Enable eager execution of tf.functions.
# tf.config.experimental_run_functions_eagerly(True)

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
# sess  = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# K.set_session(sess)

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print('+++++++++++', physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


import numpy as np
import cv2
from sklearn.metrics import pairwise_distances_argmin_min
import cv2
# from keras.utils import tree



def extract_keypoints_from_contour(contour, num_keypoints=10):
    # Calculate the total length of the contour
    # contour = np.array(contour)
    # contour = contour[0]
    len_countour = []
    for index in range(len(contour)):
        contour_length = cv2.arcLength(contour[index], True)
        len_countour.append(contour_length)
    index_max = len_countour.index(max(len_countour))
    # print(index_max)
    contour = contour[index_max]
    # T�nh d? d�i c?a contour


    # Initialize variables
    keypoints = []
    index = np.linspace(0, len(contour), num_keypoints, dtype=int, endpoint=False)
    # print(index)
    # print(index)
    for i in index:
        keypoints.append(contour[i][0])
    return keypoints

    return keypoints
# @tf.function
def extract_keypoints(mask):
    # Extract contours from the mask
    feature_map = mask[:, :, 0]
    feature_map = feature_map.numpy()
    #############################################################HUNG SUA#############################
    y_pred2 = np.where(feature_map > 0.55, 1, 0)
     #############################################################HUNG SUA#############################
    y_pred2 = np.uint8(y_pred2)
    y_true_f = y_pred2[:, :] * 255
    contours, _ = cv2.findContours(y_true_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Extract keypoints from contours
    if len(contours) > 0 :
        keypoints = extract_keypoints_from_contour(contours, num_keypoints=args.keypoint)
        return np.array(keypoints)
    else:
        return np.array([])


##############################################################################
def euclidean_distance_numpy(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point1 - point2)
    return distance


#############################################################################


#############################################################################
def active_contour_euler_elastica_loss(y_true, y_pred):
    # Ensure y_true and y_pred have batch dimensions
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=0)
    if len(y_pred.shape) == 3:
        y_pred = tf.expand_dims(y_pred, axis=0)
    
    # Ensure the shapes match
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Compute the gradients of the predicted segmentation map
    grad_y_pred = tf.image.sobel_edges(y_pred)
    grad_y_true = tf.image.sobel_edges(y_true)
    
    # Extract gradients in x and y directions for y_pred
    grad_x_pred = grad_y_pred[..., 0]
    grad_y_pred = grad_y_pred[..., 1]
    
    # Extract gradients in x and y directions for y_true
    grad_x_true = grad_y_true[..., 0]
    grad_y_true = grad_y_true[..., 1]
    
    # Compute the magnitude of the gradients
    grad_magnitude_pred = tf.sqrt(grad_x_pred**2 + grad_y_pred**2 + 1e-8)
    grad_magnitude_true = tf.sqrt(grad_x_true**2 + grad_y_true**2 + 1e-8)
    
    # Euler Term: Encourages smooth contours
    euler_term = tf.reduce_mean(tf.abs(grad_x_pred - grad_x_true) + tf.abs(grad_y_pred - grad_y_true))
    
    # Elastic Term: Ensures the contour closely matches the object boundaries
    elastic_term = tf.reduce_mean(tf.abs(y_pred - y_true))
    
    # Combined Euler Elastic Active Contour Loss
    loss = euler_term + elastic_term
    
    return loss



#########################################################################

def active_contour_euler_elastica_loss_bath(y_true, y_predict):

    batch, _, _, _ = y_true.shape
    loss_batch = []
    for ind in range(batch):
        maskimage_true = y_true[ind, :, :, :]
        maskimage_predict = y_predict[ind, :, :, :]

        # Match keypoints
        # Hàm pairwise_distances_argmin_min(X, Y) sẽ tính toán khoảng cách từ mỗi điểm trong X đến điểm tham chiếu gần nhất trong Y
        #matched_indices: Mảng chứa chỉ số của điểm tham chiếu gần nhất trong Y đối với mỗi điểm trong X
        if maskimage_true.shape[0]>0 and maskimage_predict.shape[0]>0:

            AC_loss=active_contour_euler_elastica_loss(maskimage_true,maskimage_predict)
            #print(AC_loss)
        else:
            AC_loss=0

        loss_batch.append(np.float32(AC_loss))
    # loss_batch = tf.convert_to_torch(loss_batch)

    loss_batch = tf.convert_to_tensor(loss_batch)
    return loss_batch

def read_txt(path_train, label):
    data = []
    for indexs_p in path_train:
        file1 = open(indexs_p, 'r')
        Lines = file1.readlines()
        count = 0
        # Strips the newline character
        for line in Lines:
            count += 1
            lines = line.strip().split(' ')
            if int(lines[-1]) == label:
                name = lines[0].split('.')[0]
                data.append(name)
            if label == 8:
                name = lines[0].split('.')[0]
                data.append(name)
    return data



np.random.seed(42)
tf.random.set_seed(42)

parser = argparse.ArgumentParser("simple_example")
#parser.add_argument("--keypoint", default = 1000, help="Number of keypoint", type=int)
parser.add_argument("--keypoint", default = 100, help="Number of keypoint", type=int)
parser.add_argument("--epoch", default = 100, help="Number of keypoint", type=int)
##########################################THAY DOI NHAN O DAY####################################################################################
parser.add_argument("--label", default = 7, help="Number of keypoint", type=int)
##########################################THAY DOI NHAN O DAY####################################################################################

#### Nếu default = 8 thì là chạy phân lớp binary (là hoặc không là dữ liệu của khối U) của toàn bộ OTU
## Nếu default = 0 thì lớp 0
## Nếu default = 1 thì lớp 1
###.....
## Nếu default = 7 thì lớp 7
## chạy từng lớp 1 để lấy kết quả

args = parser.parse_args()
print(args.keypoint)

# # Using pytorch
# np.random.seed(42)
# torch.manual_seed(42)
IMAGE_SIZE = 256  # Image size
EPOCHS = args.epoch       # Epochs
BATCH = 4         # Batch size
LR = 1e-4         # Learning rate 10^-4 = 0.0001


# Image path in google drive
PATH = "/media/DATA1/hunglv_data/CombineLoss/OTU_2d/"
path_train = '/media/DATA1/hunglv_data/CombineLoss/OTU_2d/train_cls.txt'
path_val = '/media/DATA1/hunglv_data/CombineLoss/OTU_2d/val_cls.txt'
path_train_all = [path_train, path_val]
data_by_class = read_txt(path_train_all, args.label)
print(data_by_class)
print(len(data_by_class))

# Add file_path with corresponding folder
images_arr = []
annotations_arr = []
get_class = 1
# filter by class



# for i in range(0, 1469):
#     path = "/home/oem/oanh/Keypoint/OTU/"
#     x = path + "/images/" + str(i+1) + ".JPG"
#     y = path + "/annotations/" + str(i+1) + ".PNG"
#     images_arr.append(x)
#     annotations_arr.append(y)

# data by class
for name in data_by_class:
    path = "/media/DATA1/hunglv_data/CombineLoss/OTU_2d/"
    x = path + "/images/" + name + ".JPG"
    y = path + "/annotations/" + name + ".PNG"
    images_arr.append(x)
    annotations_arr.append(y)


def load_data(path, split=0.1):
    # images = sorted(glob(os.path.join(path, "images/*")))
    # masks = sorted(glob(os.path.join(path, "annotations/*")))
    images = sorted(images_arr)
    masks = sorted(annotations_arr)

    total_size = len(images)                # 1469 total images
    valid_size = int(split * total_size)    # 146 images for validation
    test_size = int(split * total_size)     # 146 images for test / 1177 images for training

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42) # Divide images to training and validation set
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)  # Divide masks to training and validation set

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)  # Divide images to training and testing set
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)  # Divide masks to training and testing set

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y) # _x for ovarian ultrasound images, _y for ovarian's segmentation ground truth images

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x/255.0
    return x
def read_image_1(path):
    # path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x/255.0
    return x
# def read_image(path):
#     path = path.decode()
#     x = cv2.imread(path, cv2.IMREAD_COLOR)

#     if x is None:
#         # Xử lý trường hợp không thể đọc ảnh
#         raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {path}")

#     # Kiểm tra kích thước ảnh trước khi resize
#     if x.shape[0] == 0 or x.shape[1] == 0:
#         # Xử lý trường hợp ảnh có kích thước không hợp lý
#         raise ValueError(f"Kích thước ảnh không hợp lý: {x.shape}")

#     x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
#     x = x / 255.0
#     return x


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x/255.0
    (thresh, x) = cv2.threshold(x, 0, 1, cv2.THRESH_BINARY)
    x = np.expand_dims(x, axis=-1)
    return x

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


# X, Y là đường dẫn (file_path) dẫn đến đường dẫn của ảnh và mask tương ứng
def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64]) # Convert to json with tf.float64 type
    x.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3]) # Resize image to 256x256 -> RGB
    y.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1]) # Resize mask to 256x256  -> Gray scale
    return x, y
def tf_dataset(x, y, batch=4):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse) # Tiền xử lý map & batch
    dataset = dataset.batch(batch)
    # dataset = dataset.repeat()
    # dataset = dataset.map(Augment())
    return dataset


(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(PATH)
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

print("Training data: ", len(train_y))
print("Validation data: ", len(valid_x))
print("Testing data: ", len(test_x))

train_dataset = tf_dataset(train_x, train_y, batch=BATCH)

valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH)


import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D

# Custom layer to use tf.nn.silu
class SiLUActivationLayer(L.Layer):
    def call(self, inputs):
        return tf.nn.silu(inputs)


# Thực hiện một lớp convolution với num_filters filters và kích thước kernel là 3x3
# padding="same" được sử dụng để đảm bảo đầu ra có kích thước bằng với đầu vào
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

input = (256, 256, 3)
model = build_vgg19_attention_unet(input)
model.summary() # Sử dụng hàm summary() để xem cấu trúc chi tiết của mô hình


import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy, BinaryCrossentropy
import pandas

beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1e-15

''' Chỉ số đánh giá hiệu suất của việc phân đoạn hình ảnh'''
def dice_coef(y_true, y_pred):
    """
    Dice Coefficient
    """
    # Sử dụng tf.keras.layers.Flatten để làm phẳng tensor
    flatten = tf.keras.layers.Flatten()
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)

    # Sử dụng tf.reduce_sum thay cho K.sum
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + tf.keras.backend.epsilon()) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + tf.keras.backend.epsilon())

''' Tính toán số TRUE POSITIVE - Được biết đến là true positive rate (tỷ lệ dương tính thực sự)'''
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

"""Tính toán số TRUE NEGATIVE - được biết đến là true negative rate (tỷ lệ âm tính thực sự)"""
def specificity( y_true, y_pred):
    true_negatives = K.sum(
        K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

'''Chuyển đổi dự đoán từ đầu ra của mạng neural network thành logits, giúp áp dụng các phép toán mất mát dễ dàng hơn'''
def convert_to_logits(y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                              1 - tf.keras.backend.epsilon())
    return tf.math.log(y_pred / (1 - y_pred))

'''Tính toán mất mát entropy chéo có trọng số dựa trên logits và nhãn'''
def weighted_cross_entropyloss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                    labels=y_true,
                                                    pos_weight=pos_weight)
    return tf.reduce_mean(loss)

'''phương pháp giảm trọng số của các trường hợp dễ dự đoán đúng (đã được dự đoán đúng một cách chắc chắn)
 và tập trung vào các trường hợp khó dự đoán đúng'''
def focal_loss_with_logits( logits, labels, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * labels
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - labels)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b

def focal_loss( y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                              1 - tf.keras.backend.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))

    loss = focal_loss_with_logits(logits=logits, labels=y_true,
                                        alpha=alpha, gamma=gamma, y_pred=y_pred)

    return tf.reduce_mean(loss)

'''Thực hiện softmax trên một ma trận với chiều sâu, giúp cân bằng giá trị đầu ra của mô hình'''
def depth_softmax( matrix):
    sigmoid = lambda x: 1 / (1 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
    return softmax_matrix

'''Thực hiện softmax trên một ma trận với chiều sâu, giúp cân bằng giá trị đầu ra của mô hình'''
def generalized_dice_coefficient(y_true, y_pred):
    # smooth = 1.
    smooth = 1.0  # Đảm bảo biến smooth được định nghĩa
    # Sử dụng tf.keras.layers.Flatten để làm phẳng tensor
    flatten = tf.keras.layers.Flatten()
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)

    # Sử dụng tf.reduce_sum thay cho K.sum
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss( y_true, y_pred):
    loss = 1 - generalized_dice_coefficient(y_true, y_pred)
    return loss

# Kết hợp FOCAL và DICE
def focal_dice_loss( y_true, y_pred):
  loss = focal_loss(y_true, y_pred)+dice_loss(y_true, y_pred)
  return loss

# Kết hợp CROSS-ENTROPY và DICE
def bce_dice_loss( y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + \
            dice_loss(y_true, y_pred)
    return loss / 2.0
'''
- Tính toán precision và recall dựa trên các giá trị thực và dự đoán.
- Tính số lượng true positives, false positives và false negatives,
sau đó tính toán precision và recall dựa trên các giá trị này.
'''
def confusion(y_true, y_pred):
    # smooth = 1
    smooth = 1.0  # Đảm bảo biến smooth được định nghĩa

    y_pred_pos = tf.clip_by_value(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = tf.clip_by_value(y_true, 0, 1)
    y_neg = 1 - y_pos
    
    tp = tf.reduce_sum(y_pos * y_pred_pos)
    fp = tf.reduce_sum(y_neg * y_pred_pos)
    fn = tf.reduce_sum(y_pos * y_pred_neg)
    
    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return prec, recall

def true_positive(y_true, y_pred):
    # smooth = 1
    smooth = 1.0  # Đảm bảo biến smooth được định nghĩa

    y_pred_pos = tf.round(tf.clip_by_value(y_pred, 0, 1))
    y_pos = tf.round(tf.clip_by_value(y_true, 0, 1))
    
    tp = (tf.reduce_sum(y_pos * y_pred_pos) + smooth) / (tf.reduce_sum(y_pos) + smooth)
    
    return tp

def true_negative(y_true, y_pred):
    smooth = 1.0  # Đảm bảo biến smooth được định nghĩa

    y_pred_pos = tf.round(tf.clip_by_value(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = tf.round(tf.clip_by_value(y_true, 0, 1))
    y_neg = 1 - y_pos
    
    tn = (tf.reduce_sum(y_neg * y_pred_neg) + smooth) / (tf.reduce_sum(y_neg) + smooth)
    
    return tn

# phương pháp đo độ giống nhau giữa hai tập hợp - TP TN
def tversky_index(y_true, y_pred):
    y_true_pos = tf.keras.layers.Flatten()(y_true)
    y_pred_pos = tf.keras.layers.Flatten()(y_pred)
    
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    
    alpha = 0.3
    smooth = 1.0  # Đảm bảo biến smooth được định nghĩa
    
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss( y_true, y_pred):
    return 1 - tversky_index(y_true, y_pred)

# Hàm mất mát FOCAL dựa tren Tversky
def focal_tversky( y_true, y_pred):
    pt_1 = tversky_index(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)

# Tính mất mát log-cosh dựa trên mất mát Dice
def log_cosh_dice_loss( y_true, y_pred):
    x = dice_loss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

# Tính chỉ số tương đồng Jacard và mất mát Jacard tương ứng
def jacard_similarity( y_true, y_pred):
    """
     Intersection-Over-Union (IoU), also known as the Jaccard Index
    """
    # Sử dụng tf.keras.layers.Flatten để làm phẳng tensor
    flatten = tf.keras.layers.Flatten()
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    custom_threshold = 0.55

    # Apply threshold to convert probabilities to binary (0 or 1)
    #y_pred_f = (y_pred_f >= custom_threshold).astype(int)
    #y_pred_f = (y_pred_f >= custom_threshold, tf.int32)
    y_pred_f = tf.cast(y_pred_f >= custom_threshold, tf.float32)
    # Sử dụng tf.reduce_sum thay cho K.sum

    # Ensure both tensors are the same type
    #y_true_f = tf.cast(y_true_f, tf.float32)
    #y_pred_f = tf.cast(y_pred_f, tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
    
    return intersection / union

def jacard_loss(y_true, y_pred):
    """
      Intersection-Over-Union (IoU), also known as the Jaccard loss
    """
    return 1 - jacard_similarity(y_true, y_pred)

# Tính chỉ số tương đồng Jacard và mất mát Jacard tương ứng
def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index (SSIM) loss
    """
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1)

def unet3p_hybrid_loss(y_true, y_pred):
    """
    Hybrid loss proposed in UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
    """
    focal_loss1 = focal_loss(y_true, y_pred)
    ms_ssim_loss1 = ssim_loss(y_true, y_pred)
    jacard_loss1 = jacard_loss(y_true, y_pred)
    diceloss = dice_loss( y_true, y_pred)
    return (focal_loss1 + ms_ssim_loss1 + jacard_loss1)/3

def hybrid_loss( y_true, y_pred):
    bce_loss =binary_crossentropy(y_true, y_pred)
    dice_loss1 = dice_loss(y_true, y_pred)
    jacard_loss1 = jacard_loss(y_true, y_pred)
    return (bce_loss+ jacard_loss1+dice_loss1)/3


def binary_dice(y_true, y_pred): # xap xi hybrid_loss
    return 0.5 *binary_crossentropy(y_true, y_pred) + dice_loss( y_true, y_pred)
#[0.22297033667564392, 0.7812317609786987, 0.6458732485771179, 0.8226447701454163, 0.7904954552650452]

'''
ính toán mất mát Focal Loss, một biến thể của Cross Entropy Loss được thiết kế để tập trung vào các
mẫu khó phân loại bằng cách tăng cường trọng số của chúng.
'''
def FocalLoss(y_true, y_pred):

    # alpha = 0.3
    # gamma = 2
    alpha = 0.26
    gamma = 2.3
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    # Sử dụng tf.keras.layers.Flatten để làm phẳng tensor
    flatten = tf.keras.layers.Flatten()
    inputs = flatten(y_pred)
    targets = flatten(y_true)

    # Sử dụng tf.keras.losses.binary_crossentropy thay cho K.binary_crossentropy
    BCE = tf.keras.losses.binary_crossentropy(targets, inputs)
    BCE_EXP = tf.exp(-BCE)
    
    # Sử dụng tf.math.pow thay cho K.pow
    focal_loss = tf.reduce_mean(alpha * tf.math.pow((1 - BCE_EXP), gamma) * BCE)

    return focal_loss

#############################################################################################
##############################################################################################
###########################################################################################
def ve_len_anh(image,day_point,color,mauchu):
# Create a blank image

    # Load points from file
    points = day_point
    # Draw points on the image
    for idx, point in enumerate(points):
        draw_points(image, point,idx,color, mauchu)

# Draw points on an image
def draw_points(image, point, idx, color, mauchu, radius=1, thickness=-1):
    cv2.circle(image, (int(point[0]), int(point[1])), radius, color, thickness)
    #if (idx%10==0):
    #    cv2.putText(image, f"{idx}", (int(point[0]), int(point[1]) - 2), 
    #        cv2.FONT_HERSHEY_SIMPLEX, 0.2, mauchu, 1, cv2.LINE_AA)
############################################################################################



def joint_loss1(y_true, y_pred):

    #print('=========================active_contour_loss=====================')
    #print(y_true[1])
    #print(y_true[1].shape)

    #binary_image = tf.cast(y_true[3] > 0.5, tf.uint8) * 255
    # Squeeze the last dimension to make it compatible with OpenCV
    #binary_image = np.squeeze(binary_image)
    #cv2.imwrite('images_debug/anh_groundtruthcontour_hung3.png', binary_image)
    #print('=========================active_contour_loss=====================')
    #exit()

    focal_loss1 = FocalLoss(y_true, y_pred)
    ms_ssim_loss1 = ssim_loss(y_true, y_pred)
    jacard_loss1 = jacard_loss(y_true, y_pred)
    # loss1 = (focal_loss1 + ms_ssim_loss1 + jacard_loss1) / 3

    #keypoint_l = calculate_keypoint_loss(y_true, y_pred)
    keypoint_l = active_contour_euler_elastica_loss_bath(y_true, y_pred)
   

    loss =  (focal_loss1 + ms_ssim_loss1 + jacard_loss1 + keypoint_l)/4
    # print('------------',loss)
    return loss

# Tính toán mất mát Cross Entropy Loss nhị phân giữa y_true và y_pred
def bce_loss(y_true, y_pred):
 return binary_crossentropy(y_true, y_pred)


# Loss new


import tensorflow as tf
from tensorflow.keras import backend as K
###########################################HUNG MOI THEM###############################################
from tensorflow.keras.callbacks import ModelCheckpoint
###########################################HUNG MOI THEM###############################################

# Hàm tính IoU với ngưỡng 0.55
def iou_threshold_055(y_true, y_pred):
    # Chuyển đổi giá trị dự đoán thành nhị phân với ngưỡng 0.55
    threshold = 0.55
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # Tính giao (intersection) và hợp (union)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection

    # Tránh chia cho 0
    iou = intersection / (union + K.epsilon())
    return iou

# Precision metric dựa trên IoU > 0.55
def precision_iou_055(y_true, y_pred):
    iou = iou_threshold_055(y_true, y_pred)
    return tf.cast(iou > 0.55, tf.float32)

# Recall metric dựa trên IoU > 0.55
def recall_iou_055(y_true, y_pred):
    iou = iou_threshold_055(y_true, y_pred)
    return tf.cast(iou > 0.55, tf.float32)

######################################################HUNG VIET THEM##############################################
checkpoint_callback = ModelCheckpoint(
    filepath='/media/DATA1/hunglv_data/CombineLoss/output_OTU_2D_hung_ACE_loss_4_055_7/best_vgg19_attention_unet.h5',  # Path to save the model
    monitor='val_precision',  # Monitor validation precision
    save_best_only=True,  # Save only the best model
    save_weights_only=False,  # Save the entire model (set to True to save only weights)
    mode='max',  # 'max' because we want to maximize precision
    verbose=1  # Print messages when saving
)
######################################################HUNG VIET THEM##############################################


opt = tf.keras.optimizers.Nadam(LR)
metrics = [dice_coef, jacard_similarity, Recall(), Precision()]

model.compile(loss=joint_loss1,optimizer=opt, metrics=metrics, run_eagerly=True)

# model.compile(loss=calculate_keypoint_loss,optimizer=opt, metrics=metrics, run_eagerly=True)
# experimental_run_tf_function = False,
#################################################HUNG DONG LAI############################################
#callbacks = [
#    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
#    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
#]
#################################################HUNG DONG LAI############################################
train_steps = len(train_x)//BATCH
valid_steps = len(valid_x)//BATCH

#if len(train_x) % BATCH != 0:
#    train_steps += 1
#if len(valid_x) % BATCH != 0:
#    valid_steps += 1

iou_threshold = 0.55
with tf.device('/gpu:0'):
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps, callbacks=[checkpoint_callback]
        
    )

test_dataset = tf_dataset(test_x, test_y, batch=BATCH)

test_steps = (len(test_x)//BATCH)
if len(test_x) % BATCH != 0:
    test_steps += 1

model.evaluate(test_dataset, steps=test_steps)


iou_threshold = 0.55
all_loss = []
for index in range(len(test_x)):
    # print(test_x[index])
    img_name = test_x[index].split('/')[-1]
    x = read_image_1(test_x[index])
    # y = read_mask(test_y[index])
    y_true = cv2.imread(test_y[index], cv2.IMREAD_GRAYSCALE)
    y_true = cv2.resize(y_true, (256, 256))

    #y_pred2 = model.predict(np.expand_dims(x, axis=0), iou=iou_threshold)[0]
    y_pred2 = model.predict(np.expand_dims(x, axis=0))[0]

    b = model.predict(np.expand_dims(x, axis=0))
    ########################################################hung viet################################################################
    y_pred2 = np.where(y_pred2 > 0.55, 1, 0)
    ########################################################hung viet################################################################
    y_pred2 = np.uint8(y_pred2)
    y_pred2 = y_pred2[:, : , 0]*255
    y_predict = y_pred2
    cv2.imwrite('/media/DATA1/hunglv_data/CombineLoss/output_OTU_2D_hung_ACE_loss_4_055_7/{}'.format(img_name), y_pred2)
    # sum_img = np.sum(y_predict)
    # img = np.zeros((256, 256, 3), dtype=np.uint8)
    # if sum_img > 0:
    #     loss, keypoints_true, keypoints_predict, contours_true, contours_predict = calculate_keypoint_loss(y_true, y_predict)
    #     all_loss.append(loss)
    #     # print('loss: ', loss)
    #     # visual point
    #     for point in keypoints_true:
    #         x, y = point
    #         cv2.circle(img, (x, y), 2, (0, 0, 255), -1)  # Draw a green circle at each keypoint\
    #     for point in keypoints_predict:
    #         x, y = point
    #         cv2.circle(img, (x, y), 2, (0, 255, 255), -1)  # Draw a green circle at each keypoint
    #
    #     # contours_ = contours_true + contours_predict
    #     cv2.drawContours(img, contours_true, -1, (255, 255, 255), 1)
    #
    #     cv2.drawContours(img, contours_predict, -1, (255, 255, 0), 1)
    #     cv2.imwrite('visual/visual_{}'.format(img_name), img)
    #     # Display the image with keypoints
        # cv2.imshow('Image with keypoints', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # contours, _ = cv2.findContours(y_pred2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imwrite('t.jpg', y_pred2)
    # inference_time = time.time() - start_time


# avg_loss = sum(all_loss)/len(all_loss)
# name_weight = path_weight.split('/')[-1]
# print('Number of sample: ', len(all_loss))
# print("Loss of model {} is: ".format(name_weight), avg_loss)
# loss_c = [name_weight, avg_loss]
# loss_each_epoch.append(loss_c)
#
# print(loss_each_epoch)






pandas.DataFrame(history.history).to_csv("/media/DATA1/hunglv_data/CombineLoss/output_OTU_2D_hung_ACE_loss_4_055_7/history_{}_{}.csv".format(str(args.keypoint), str(args.label)))
model.save_weights('/media/DATA1/hunglv_data/CombineLoss/output_OTU_2D_hung_ACE_loss_4_055_7/model_{}_{}.weights.h5'.format(str(args.keypoint), str(args.label)))



def plot_chart(history):
    dice = history.history['dice_coef']
    val_dice = history.history['val_dice_coef']
    IoU= history.history['jacard_similarity']
    val_IoU = history.history['val_jacard_similarity']
    Recall= history.history['recall']
    val_recall = history.history['val_recall']
    Precision= history.history['precision']
    val_Precision = history.history['val_precision']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(dice))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, dice, 'r', label='Training dice')
    plt.plot(epochs, val_dice, 'b', label='Validation dice')
    plt.plot(epochs, IoU, 'maroon', label='jacard_similarity')
    plt.plot(epochs, val_IoU, 'orange', label='val_jacard_similarity')
    plt.plot(epochs, Recall, 'pink', label='recall')
    plt.plot(epochs, val_recall, 'brown', label='val_recall')
    plt.plot(epochs, Recall, 'violet', label='precision')
    plt.plot(epochs, val_recall, 'tan', label='val_precision')
    plt.plot(epochs, loss, 'g', label='Training Loss')
    plt.plot(epochs, val_loss, 'y', label='Validation Loss')

    plt.title('Metrics and Loss')
    plt.legend(loc=0)
    plt.savefig('/media/DATA1/hunglv_data/CombineLoss/output_OTU_2D_hung_ACE_loss_4_055_7/model_and_loss_{}_{}.png'.format(str(args.keypoint), str(args.label)))
    plt.show()

plot_chart(history)

def plot_chart(history):

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))
    plt.figure(figsize=(10, 6))
    plt.xlabel('number of epochs')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, 'g', label='Training Loss')
    plt.plot(epochs, val_loss, 'y', label='Validation Loss')

    plt.title('model loss')
    plt.legend(loc=0)
    plt.savefig('/media/DATA1/hunglv_data/CombineLoss/output_OTU_2D_hung_ACE_loss_4_055_7/model_loss_{}_{}.png'.format(str(args.keypoint), str(args.label)))
    plt.show()


plot_chart(history)

def plot_chart(history):
    dice = history.history['dice_coef']
    val_dice = history.history['val_dice_coef']
    IoU= history.history['jacard_similarity']
    val_IoU = history.history['val_jacard_similarity']


    epochs = range(len(dice))
    plt.figure(figsize=(10, 6))
    plt.xlabel('number of epochs')
    plt.ylabel('metric')
    plt.plot(epochs, dice, 'r', label='Training dice')
    plt.plot(epochs, val_dice, 'b', label='Validation dice')
    plt.plot(epochs, IoU, 'maroon', label='jacard_similarity')
    plt.plot(epochs, val_IoU, 'orange', label='val_jacard_similarity')

    plt.title('Model Metrics')
    plt.legend(loc=0)
    plt.savefig('/media/DATA1/hunglv_data/CombineLoss/output_OTU_2D_hung_ACE_loss_4_055_7/model_metrics_{}_{}.png'.format(str(args.keypoint), str(args.label)))
    plt.show()

plot_chart(history)


