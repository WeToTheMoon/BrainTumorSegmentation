import os
import random

import elasticdeform
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv3D, concatenate, Conv3DTranspose, LeakyReLU
from keras.models import Model
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import affine_transform
from tensorflow import keras
from tensorflow_addons.layers import InstanceNormalization

from augmentations import brightness
from loader import load_img, imageLoader_val
from metric import dice_coef
from loss import dice_coef_loss


def binary_elastic(x, y):
    [x_el, y_el] = elasticdeform.deform_random_grid(
        [x, y], sigma=2, axis=[(0, 1, 2), (0, 1, 2)], order=[1, 0], mode='constant')

    return x_el, y_el


def rotation(x, y):
    """
    Rotate a 3D image with alfa, beta and gamma degree respect the axis x, y and z respectively.
    The three angles are chosen randomly between 0-30 degrees
    """

    alpha, beta, gamma = np.random.random_sample(3) * np.pi / 10
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])

    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])

    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])

    R_rot = np.dot(np.dot(Rx, Ry), Rz)

    a, b = 0.8, 1.2
    alpha, beta, gamma = (b - a) * np.random.random_sample(3) + a
    R_scale = np.array([[alpha, 0, 0],
                        [0, beta, 0],
                        [0, 0, gamma]])

    R = np.dot(R_rot, R_scale)
    X_rot = np.empty_like(x)
    y_rot = np.empty_like(y)
    for b in range(x.shape[0]):
        for channel in range(x.shape[-1]):
            X_rot[b, :, :, :, channel] = affine_transform(
                x[b, :, :, :, channel], R, offset=0, order=1, mode='constant')
    for b in range(x.shape[0]):
        y_rot[b, :, :, :] = affine_transform(
            y[b, :, :, :], R, offset=0, order=0, mode='constant')

    return X_rot, y_rot


def binary_combine_aug(x, y):
    x_new, y_new = x, y

    if np.random.randint(0, 10) < 3:
        x_new, y_new = binary_elastic(x_new, y_new)

    if np.random.randint(0, 10) < 3:
        x_new, y_new = brightness(x_new, y_new)

    if np.random.randint(0, 10) < 3:
        x_new, y_new = rotation(x_new, y_new)

    return x_new, y_new


def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    # keras needs the generator infinite, so we will use while true
    while True:
        batch_start = 0
        batch_end = batch_size

        temp1 = list(zip(img_list, mask_list))
        random.shuffle(temp1)
        img_list, mask_list = zip(*temp1)
        img_list, mask_list = list(img_list), list(mask_list)

        L = len(img_list)

        while batch_start < L:
            limit = min(batch_end, L)

            x = load_img(img_dir, img_list[batch_start:limit])
            y = load_img(mask_dir, mask_list[batch_start:limit])

            binary_combine_aug(x, y)

            batch_start += batch_size
            batch_end += batch_size

            yield x, y  # a tuple with two numpy arrays with batch_size samples


def binary_double_conv_block(x, n_filters, activation='relu'):
    x = InstanceNormalization()(x)
    x = Conv3D(n_filters, 3, padding="same", activation=activation)(x)
    x2 = InstanceNormalization()(x)
    x2 = Conv3D(n_filters, 3, padding="same", activation=activation)(x2)
    x2 = InstanceNormalization()(x2)
    return x2


train_img_dir = r"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\train\images"
train_mask_dir = r"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\train\masks"

val_img_dir = r"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\val\images"
val_mask_dir = r"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\val\masks"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader_val(val_img_dir, val_img_list,
                                  val_mask_dir, val_mask_list, batch_size)

LR = 0.0003
optim = keras.optimizers.Adam(LR)

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size


def simple_unet_model(
        img_height: int, img_width: int,
        img_depth: int, img_channels: int,
        num_classes: int, channels: int, activation="relu"):
    # Build the model
    inputs = Input((img_height, img_width, img_depth, img_channels))
    s = inputs

    # Contraction pat
    c2 = binary_double_conv_block(s, channels * 2)
    p2 = Conv3D(channels * 2, (3, 3, 3), strides=(2, 2, 2), activation=activation, padding='same')(c2)

    c3 = binary_double_conv_block(p2, channels * 4)
    p3 = Conv3D(channels * 4, (3, 3, 3), strides=(2, 2, 2), activation=activation, padding='same')(c3)

    c4 = binary_double_conv_block(p3, channels * 8)
    p4 = Conv3D(channels * 8, (3, 3, 3), strides=(2, 2, 2), activation=activation, padding='same')(c4)

    c5 = binary_double_conv_block(p4, channels * 10)

    # Expansive path
    u6 = Conv3DTranspose(channels * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = binary_double_conv_block(u6, channels * 8)

    u7 = Conv3DTranspose(channels * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = binary_double_conv_block(u7, channels * 4)

    u8 = Conv3DTranspose(channels * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = binary_double_conv_block(u8, channels * 2)

    outputs = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(c8)

    return Model(inputs=[inputs], outputs=[outputs])


n_channels = 20
model = simple_unet_model(128, 128, 128, 4, 1, n_channels)

model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])

checkpoint_path = r'C:\Users\kesch\OneDrive\Documents\Deeplearning\seg_weights\binary1.hdf5'
callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)

# model.load_weights(checkpoint_path)

# history = model.fit(train_img_datagen,
#                     steps_per_epoch=steps_per_epoch,
#                     epochs=1000,
#                     verbose=1,
#                     validation_data=val_img_datagen,
#                     validation_steps=val_steps_per_epoch,
#                     callbacks=[callback])
