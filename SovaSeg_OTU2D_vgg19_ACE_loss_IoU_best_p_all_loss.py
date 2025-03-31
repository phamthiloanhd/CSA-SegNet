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
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU, Conv2D, UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2,ResNet50, VGG19

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K
tf.compat.v1.enable_eager_execution()
import argparse
import numpy as np
import cv2
from sklearn.metrics import pairwise_distances_argmin_min
import tensorflow.keras.layers as L
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D
from keras.losses import binary_crossentropy, BinaryCrossentropy
import pandas

from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_vgg19_attention_unet
from visual import plot_chart_training, plot_chart_loss, plot_chart_metrics
from loss import joint_loss, dice_coef, jacard_similarity

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # run the code on a specified GPU
# Kiểm tra danh sách các GPU có sẵn
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available")
else:
    print("GPU is not available")



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

def load_data(images_arr,annotations_arr, split=0.1):
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



np.random.seed(42)
tf.random.set_seed(42)

parser = argparse.ArgumentParser("simple_example")
#parser.add_argument("--keypoint", default = 1000, help="Number of keypoint", type=int)
parser.add_argument("--keypoint", default = 100, help="Number of keypoint", type=int)
parser.add_argument("--epoch", default = 100, help="Number of keypoint", type=int)
##########################################THAY DOI NHAN O DAY####################################################################################
parser.add_argument("--label", default = 7, help="Number of keypoint", type=int)

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
# data by class
for name in data_by_class:
    path = "/media/DATA1/hunglv_data/CombineLoss/OTU_2d/"
    x = path + "/images/" + name + ".JPG"
    y = path + "/annotations/" + name + ".PNG"
    images_arr.append(x)
    annotations_arr.append(y)


(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(images_arr,annotations_arr)
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

print("Training data: ", len(train_y))
print("Validation data: ", len(valid_x))
print("Testing data: ", len(test_x))

train_dataset = tf_dataset(train_x, train_y, batch=BATCH)

valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH)


input = (256, 256, 3)
model = build_vgg19_attention_unet(input)
model.summary() # Sử dụng hàm summary() để xem cấu trúc chi tiết của mô hình

beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1e-15

checkpoint_callback = ModelCheckpoint(
    filepath='/media/DATA1/hunglv_data/CombineLoss/output_OTU_2D_hung_ACE_loss_4_055_7/best_vgg19_attention_unet.h5',  # Path to save the model
    monitor='val_precision',  # Monitor validation precision
    save_best_only=True,  # Save only the best model
    save_weights_only=False,  # Save the entire model (set to True to save only weights)
    mode='max',  # 'max' because we want to maximize precision
    verbose=1  # Print messages when saving
)


opt = tf.keras.optimizers.Nadam(LR)
metrics = [dice_coef, jacard_similarity, Recall(), Precision()]

model.compile(loss=joint_loss,optimizer=opt, metrics=metrics, run_eagerly=True)

train_steps = len(train_x)//BATCH
valid_steps = len(valid_x)//BATCH

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

pandas.DataFrame(history.history).to_csv("/media/DATA1/hunglv_data/CombineLoss/output_OTU_2D_hung_ACE_loss_4_055_7/history_{}_{}.csv".format(str(args.keypoint), str(args.label)))
model.save_weights('/media/DATA1/hunglv_data/CombineLoss/output_OTU_2D_hung_ACE_loss_4_055_7/model_{}_{}.weights.h5'.format(str(args.keypoint), str(args.label)))

plot_chart_training(history)
plot_chart_loss(history)
plot_chart_metrics(history)





