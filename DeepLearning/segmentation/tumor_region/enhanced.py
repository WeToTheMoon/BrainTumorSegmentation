from keras.models import Model
from keras.layers import Input, Conv3D, concatenate, Conv3DTranspose, LeakyReLU, Dense, Multiply, MaxPooling3D, Lambda, Activation, UpSampling3D, Add, BatchNormalization, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
from tensorflow import keras
import random
import keras.backend as K
from keras.models import load_model
from numpy.random import randint
import os
from tensorflow_addons.layers import InstanceNormalization
import elasticdeform

def brightness(X, y):

    X_new = np.zeros(X.shape)
    for c in range(X.shape[-1]):
        im = X[:, :, :, c]
        gain, gamma = (1.2 - 0.8) * np.random.random_sample(2,) + 0.8
        im_new = np.sign(im)*gain*(np.abs(im)**gamma)
        X_new[:, :, :, c] = im_new

    return X_new, y

def elastic(X, y):

    [Xel, yel] = elasticdeform.deform_random_grid(
        [X, y], sigma=2, axis=[(0, 1, 2), (0, 1, 2),], order=[1, 0], mode='constant')

    return Xel, yel

def combine_aug(x, y, dec):
    Xnew, ynew = x, y
    # if dec[0] == 1:
    #     Xnew, x1new, ynew = elastic(Xnew, x1new, ynew)

    if dec[0] == 1:
        Xnew, ynew = brightness(Xnew, ynew)

    if dec[1] == 1:
        Xnew, ynew = elastic(Xnew, ynew)
    return Xnew, ynew

def load_img(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):
        if (image_name.split('.')[1] == 'npy'):
            if "image" in image_name:
                image = np.load(img_dir + '/' + image_name)
                # img_dir1 = img_dir.replace("BrainTumorSeg", "BrainTumorSegTemp")
                # image_name1 = image_name.replace(".npy", " (3).npy")
                # image1 = np.load(img_dir1 + '/' + image_name1)
                # image1 = image1.reshape(image.shape[0],image.shape[1],image.shape[2],1)
                # image = np.concatenate((image,image1), axis=3)
                images.append(image[:,:,:,:-1])
            else:
                image = np.load(img_dir + '/' + image_name)
                image = np.array(image[:,:,:,3]).reshape(image.shape[0],image.shape[1],image.shape[2],1)
                images.append(image)
            
    # images = np.array(images)
    return(images)

def global_extraction(img, mask):
    img1 = []
    mask1 = []
    for i in range(len(img)):
        img_temp = img[i]
        mask_temp = mask[i]
        or_r = randint(0, img_temp.shape[0]-47)
        or_c = randint(0, img_temp.shape[1]-47)
        img_temp = img_temp[or_r:or_r+48,or_c:or_c+48,:,:]
        mask_temp = mask_temp[or_r:or_r+48,or_c:or_c+48,:,:]
        img1.append(img_temp)
        mask1.append(mask_temp)
    img_final = np.stack((i for i in img1), axis=0)
    mask_final = np.stack((i for i in mask1), axis=0)
    return img_final, mask_final

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
            X,Y = global_extraction(X,Y)
            temp = np.zeros(2)

            for i in range(len(temp)):
                temp[i] = np.random.randint(3)

            combine_aug(X, Y, temp)

            batch_start += batch_size
            batch_end += batch_size

            yield (X, Y)  # a tuple with two numpy arrays with batch_size samples

def imageLoader_val(img_dir, img_list, mask_dir, mask_list, batch_size1):

    L = len(img_list)

    #keras needs the generator infinite, so we will use while true  
    while True:

        batch_start = 0
        batch_end = batch_size1

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            X,Y = global_extraction(X,Y)

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size1
            batch_end += batch_size1

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    if K.sum(y_true_f) == 0.0:
        if K.sum(y_pred_f) == 0.0:
            return 1.0
        else: 
            return 0.0
    else:
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def weighted_dice_loss(y_true, y_pred):
    return (1-dice_coef_multilabel(y_true, y_pred))  # taking average
    
def dice_coef_multilabel(y_true, y_pred, numLabels=3):
    dice = 0
    for index in range(numLabels):
        dice += dice_coef(y_true, y_pred)
    return dice/numLabels  # taking average

def double_conv_block(x, n_filters):
    x1 = InstanceNormalization()(x)
    x1 = Conv3D(n_filters, 3, padding="same", activation=leaky_relu)(x1)
    x2 = InstanceNormalization()(x1)
    x2 = Conv3D(n_filters, 3, padding="same", activation=leaky_relu)(x2)
    return x2

# def dice_coef(y_true, y_pred):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = np.sum(y_true_f * y_pred_f)
#     smooth = 0.000001
#     return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
####################################################

train_img_dir = r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\train\images"
train_mask_dir = r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\train\masks"

val_img_dir = r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images"
val_mask_dir = r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\masks"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

################################################################

batch_size = 2
#Change global extraction also

train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader_val(val_img_dir, val_img_list,
                              val_mask_dir, val_mask_list, batch_size)

###############################################################

LR = 0.0001
optim = keras.optimizers.Adam(LR)

def lr_time_decay(epoch, lr):
    return lr * (1 - (epoch/300))**0.9

################################################################

Metrics = [dice_coef_multilabel]

steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(train_img_list)//batch_size

################################################################
leaky_relu = LeakyReLU(alpha=0.01)
# Try decreasing

n_channels = 32

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, num_classes):
    # Build the model
    input_global = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 4))

    #Contraction path
    c2 = double_conv_block(input_global, n_channels*2)
    p2 = Conv3D(n_channels*2, 3, padding="same", activation=leaky_relu, strides = (2,2,2))(c2)

    c3 = double_conv_block(p2, n_channels*4)
    p3 = Conv3D(n_channels*4, 3, padding="same", activation=leaky_relu, strides = (2,2,2))(c3)

    c4 = double_conv_block(p3, n_channels*8)
    p4 = Conv3D(n_channels*8, 3, padding="same", activation=leaky_relu, strides = (2,2,2))(c4)

    c5 = double_conv_block(p4, n_channels*10)

    #Expansive path
    u6 = Conv3DTranspose(n_channels*8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = double_conv_block(u6, n_channels*8)

    u7 = Conv3DTranspose(n_channels*4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = double_conv_block(u7, n_channels*4)

    u8 = Conv3DTranspose(n_channels*2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = double_conv_block(u8, n_channels*2)
    
    # o1 = Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)
    # o2 = Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)
    # o3 = Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)
    # o4 = Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)

    # outputs = concatenate([o1,o2,o3,o4])

    outputs = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(c8)
    # Test unpadded convolution and strided convolution
    # Try using sigmoid (research papers)
    model = Model(inputs=[input_global], outputs=[outputs])
    model.summary()
    return model

# Test if everything is working ok.
model = simple_unet_model(48, 48, 128, 1)

#Total Loss
model.compile(optimizer=optim, loss=weighted_dice_loss, metrics=Metrics)

checkpoint_path = r'C:\Users\kesch\OneDrive\Documents\Deeplearning\seg_weights\enhanced.hdf5'
callback = ModelCheckpoint(filepath=checkpoint_path,
                           save_weights_only=True, verbose=1)
# model.load_weights(r'C:\Users\kesch\OneDrive\Documents\Deeplearning\seg_weights\enhanced.hdf5')

history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=1000,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch,
                    callbacks=[callback],
                    )

# , LearningRateScheduler(lr_time_decay, verbose = 1)