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


def elastic(x, y):
    [x_el, y_el] = elasticdeform.deform_random_grid(
        [x, y], sigma=2, axis=[(0, 1, 2), (0, 1, 2)], order=[1, 0], mode='constant')

    return x_el, y_el


def rotation_zoom3D(x, y):
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


def combine_aug(x, y, dec):
    x_new, y_new = x, y
    if dec[0] == 1:
        x_new, y_new = elastic(x_new, y_new)

    if dec[1] == 1:
        x_new, y_new = brightness(x_new, y_new)

    if dec[2] == 1:
        x_new, y_new = rotation_zoom3D(x_new, y_new)

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

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            temp = np.zeros(3)

            for i in range(len(temp)):
                temp[i] = np.random.randint(3)

            combine_aug(X, Y, temp)

            batch_start += batch_size
            batch_end += batch_size

            yield (X, Y)  # a tuple with two numpy arrays with batch_size samples


def double_conv_block(x, n_filters):
    leaky_relu = LeakyReLU(alpha=0.01)
    x = InstanceNormalization()(x)
    x = Conv3D(n_filters, 3, padding="same", activation=leaky_relu)(x)
    x2 = InstanceNormalization()(x)
    x2 = Conv3D(n_filters, 3, padding="same", activation=leaky_relu)(x2)
    x2 = InstanceNormalization()(x2)
    return x2


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


####################################################


train_img_dir = r"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\train\images"
train_mask_dir = r"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\train\masks"

val_img_dir = r"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\val\images"
val_mask_dir = r"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\val\masks"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

################################################################

batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader_val(val_img_dir, val_img_list,
                                  val_mask_dir, val_mask_list, batch_size)

###############################################################
LR = 0.0003
optim = keras.optimizers.Adam(LR)

################################################################

Metrics = [dice_coef]

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

################################################################
leaky_relu = 'relu'
# LeakyReLU(alpha=0.01)
n_channels = 20


# Try decreasing

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, IMG_DEPTH, num_classes):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, IMG_DEPTH))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    # Contraction pat
    c2 = double_conv_block(s, n_channels * 2)
    p2 = Conv3D(n_channels * 2, (3, 3, 3), strides=(2, 2, 2), activation=leaky_relu, padding='same')(c2)

    c3 = double_conv_block(p2, n_channels * 4)
    p3 = Conv3D(n_channels * 4, (3, 3, 3), strides=(2, 2, 2), activation=leaky_relu, padding='same')(c3)

    c4 = double_conv_block(p3, n_channels * 8)
    p4 = Conv3D(n_channels * 8, (3, 3, 3), strides=(2, 2, 2), activation=leaky_relu, padding='same')(c4)

    c5 = double_conv_block(p4, n_channels * 10)

    # Expansive path
    u6 = Conv3DTranspose(n_channels * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = double_conv_block(u6, n_channels * 8)

    u7 = Conv3DTranspose(n_channels * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = double_conv_block(u7, n_channels * 4)

    u8 = Conv3DTranspose(n_channels * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = double_conv_block(u8, n_channels * 2)

    outputs = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(c8)
    # Try using sigmoid (research papers)
    model = Model(inputs=[inputs], outputs=[outputs])

    # Changed dropout https://github.com/bnsreenu/python_for_microscopists/blob/master/231_234_BraTa2020_Unet_segmentation/simple_3d_unet.py
    # compile model outside of this function to make it flexible.
    model.summary()
    return model


# Test if everything is working ok.
model = simple_unet_model(128, 128, 128, 4, 1)

# Total Loss
model.compile(optimizer=optim, loss=dice_coef_loss, metrics=Metrics)

checkpoint_path = r'C:\Users\kesch\OneDrive\Documents\Deeplearning\seg_weights\binary1.hdf5'
callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
# model.load_weights(r'C:\Users\kesch\OneDrive\Documents\Deeplearning\seg_weights\binary1.hdf5')

# history = model.fit(train_img_datagen,
#                     steps_per_epoch=steps_per_epoch,
#                     epochs=1000,
#                     verbose=1,
#                     validation_data=val_img_datagen,
#                     validation_steps=val_steps_per_epoch,
#                     callbacks=[callback],
#                     )

img_num = 144

test_img = np.load(
    r"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\val\images\image_" + str(img_num) + ".npy")

test_mask = np.load(
    r"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\val\masks\mask_" + str(img_num) + ".npy")

# test_mask_argmax = np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = model.predict(test_img_input)
test_prediction = test_prediction[0, :, :, :, 0]

for i in range(128):
    n_slice = i
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(test_mask[:, :, n_slice])
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(test_prediction[:, :, n_slice])
    plt.show()
