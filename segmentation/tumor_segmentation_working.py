import os
import sys
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from loader import imageLoader, imageLoader_val_crop
from loss import dice_loss
from metric import dice_coef_multilabel, core_tumor, peritumoral_edema, enhancing_tumor
from models import multiclass_model

np.set_printoptions(threshold=sys.maxsize)

train_img_dir = r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\train\images"
train_mask_dir = r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\train\masks"

val_img_dir = r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images"
val_mask_dir = r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\masks"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

model_whole = multiclass_model(48, 48, 128, 4, 1)
model = multiclass_model(48, 48, 128, 5, 1)

model_whole.load_weights(r'C:\Users\kesch\OneDrive\Documents\Deeplearning\seg_weights\whole.hdf5')

batch_size = 8

train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
val_img_datagen = imageLoader_val_crop(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size, model_whole)

LR = 0.0003
optim = keras.optimizers.Adam(LR)

model.compile(optimizer=optim, loss=dice_loss, metrics=[dice_coef_multilabel, core_tumor,
                                                        peritumoral_edema, enhancing_tumor])

checkpoint_path = r'C:\Users\kesch\OneDrive\Documents\Deeplearning\seg_weights\combined_working.hdf5'
callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)


def uncert(img):
    # given img shape is 1, 48, 48, 128, 4
    region_based = model_whole.predict(img)
    img8 = np.concatenate((img, region_based), axis=-1)
    test_prediction8 = model.predict([img8])

    img1 = np.flip(img, axis=1)
    region_based1 = model_whole.predict(img1)
    img1 = np.concatenate((img1, region_based1), axis=-1)
    test_prediction1 = model.predict([img1])
    test_prediction1 = np.flip(test_prediction1, axis=1)

    img2 = np.flip(img, axis=2)
    region_based2 = model_whole.predict(img2)
    img2 = np.concatenate((img2, region_based2), axis=-1)
    test_prediction2 = model.predict([img2])
    test_prediction2 = np.flip(test_prediction2, axis=2)

    img3 = np.flip(img, axis=3)
    region_based3 = model_whole.predict(img3)
    img3 = np.concatenate((img3, region_based3), axis=-1)
    test_prediction3 = model.predict([img3])
    test_prediction3 = np.flip(test_prediction3, axis=3)

    img4 = np.flip(img, axis=(1, 2))
    region_based4 = model_whole.predict(img4)
    img4 = np.concatenate((img4, region_based4), axis=-1)
    test_prediction4 = model.predict([img4])
    test_prediction4 = np.flip(test_prediction4, axis=(1, 2))

    img5 = np.flip(img, axis=(2, 3))
    region_based5 = model_whole.predict(img5)
    img5 = np.concatenate((img5, region_based5), axis=-1)
    test_prediction5 = model.predict([img5])
    test_prediction5 = np.flip(test_prediction5, axis=(2, 3))

    img6 = np.flip(img, axis=(3, 1))
    region_based6 = model_whole.predict(img6)
    img6 = np.concatenate((img6, region_based6), axis=-1)
    test_prediction6 = model.predict([img6])
    test_prediction6 = np.flip(test_prediction6, axis=(1, 3))

    img7 = np.flip(img, axis=(1, 2, 3))
    region_based7 = model_whole.predict(img7)
    img7 = np.concatenate((img7, region_based7), axis=-1)
    test_prediction7 = model.predict([img7])
    test_prediction7 = np.flip(test_prediction7, axis=(1, 2, 3))

    test_prediction = (test_prediction8 + test_prediction1 + test_prediction2 + test_prediction3 + test_prediction4
                       + test_prediction5 + test_prediction6 + test_prediction7) / 8
    confidence = np.zeros((1, 48, 48, 128))
    for k in range(128):
        for j in range(48):
            for l in range(48):
                confidence[:, l, j, k] = tf.math.reduce_logsumexp(-1 * (test_prediction[0, l, j, k]))
    return confidence


def put_together(img, msk):
    img = np.expand_dims(img, axis=0)
    msk = np.expand_dims(msk, axis=0)

    a = 0
    b = 48
    c = 0
    d = 48

    img_final = np.zeros(img.shape)
    uncertainty = np.zeros((img.shape[0], img.shape[1], img.shape[2], img.shape[3]))
    for i in range((img.shape[2] // 48) + 1):
        a = 0
        b = 48
        for j in range((img.shape[1] // 48) + 1):
            img_temp = img[:, a:b, c:d, :, :]
            uncertainty[:, a:b, c:d, :] = uncert(img_temp)
            region_based = model_whole.predict(img_temp)

            img_temp = np.concatenate((img_temp, region_based), axis=-1)
            test_prediction = model.predict([img_temp])
            img_final[:, a:b, c:d, :, :] = test_prediction
            if b + 48 < img.shape[1]:
                a += 48
                b += 48
            else:
                b = img.shape[1]
                a = b - 48
        if d + 48 < img.shape[2]:
            c += 48
            d += 48
        else:
            d = img.shape[2]
            c = d - 48

    img_final = np.argmax(img_final, axis=4)
    msk = np.argmax(msk, axis=4)
    return img_final, msk, uncertainty
