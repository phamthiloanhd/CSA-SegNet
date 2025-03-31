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
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG19

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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # run the code on a specified GPU
# Kiểm tra danh sách các GPU có sẵn
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available")
else:
    print("GPU is not available")


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
        # matched_indices: Mảng chứa chỉ số của điểm tham chiếu gần nhất trong Y đối với mỗi điểm trong X
        if maskimage_true.shape[0] > 0 and maskimage_predict.shape[0] > 0:

            AC_loss = active_contour_euler_elastica_loss(maskimage_true, maskimage_predict)
            # print(AC_loss)
        else:
            AC_loss = 0

        loss_batch.append(np.float32(AC_loss))
    # loss_batch = tf.convert_to_torch(loss_batch)

    loss_batch = tf.convert_to_tensor(loss_batch)
    return loss_batch


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


# Tính chỉ số tương đồng Jacard và mất mát Jacard tương ứng
def jacard_similarity(y_true, y_pred):
    """
     Intersection-Over-Union (IoU), also known as the Jaccard Index
    """
    # Sử dụng tf.keras.layers.Flatten để làm phẳng tensor
    flatten = tf.keras.layers.Flatten()
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    custom_threshold = 0.55

    # Apply threshold to convert probabilities to binary (0 or 1)
    # y_pred_f = (y_pred_f >= custom_threshold).astype(int)
    # y_pred_f = (y_pred_f >= custom_threshold, tf.int32)
    y_pred_f = tf.cast(y_pred_f >= custom_threshold, tf.float32)
    # Sử dụng tf.reduce_sum thay cho K.sum

    # Ensure both tensors are the same type
    # y_true_f = tf.cast(y_true_f, tf.float32)
    # y_pred_f = tf.cast(y_pred_f, tf.float32)

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


# [0.22297033667564392, 0.7812317609786987, 0.6458732485771179, 0.8226447701454163, 0.7904954552650452]

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


############################################################################################


def joint_loss(y_true, y_pred):
    # print('=========================active_contour_loss=====================')
    # print(y_true[1])
    # print(y_true[1].shape)

    # binary_image = tf.cast(y_true[3] > 0.5, tf.uint8) * 255
    # Squeeze the last dimension to make it compatible with OpenCV
    # binary_image = np.squeeze(binary_image)
    # cv2.imwrite('images_debug/anh_groundtruthcontour_hung3.png', binary_image)
    # print('=========================active_contour_loss=====================')
    # exit()

    focal_loss1 = FocalLoss(y_true, y_pred)
    ms_ssim_loss1 = ssim_loss(y_true, y_pred)
    jacard_loss1 = jacard_loss(y_true, y_pred)
    # loss1 = (focal_loss1 + ms_ssim_loss1 + jacard_loss1) / 3
    # keypoint_l = calculate_keypoint_loss(y_true, y_pred)
    keypoint_l = active_contour_euler_elastica_loss_bath(y_true, y_pred)

    loss = (focal_loss1 + ms_ssim_loss1 + jacard_loss1 + keypoint_l) / 4
    # print('------------',loss)
    return loss