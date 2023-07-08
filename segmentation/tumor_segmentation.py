import os
import sys
import numpy as np
from keras.callbacks import ModelCheckpoint
from utils.loader import imageLoader_val_crop, imageLoader_crop
from utils.loss import log_cosh_dice_loss
from utils.metrics import dice_coef_multilabel, core_tumor, peritumoral_edema, enhancing_tumor
from utils.models import multiclass_model
from utils.optimizers import LH_Adam

np.set_printoptions(threshold=sys.maxsize)

train_img_dir = r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\train\images"
train_mask_dir = r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\train\masks"

val_img_dir = r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images"
val_mask_dir = r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\masks"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

binary_model = multiclass_model(128, 128, 128, 4, 1)
model = multiclass_model(48, 48, 128, 5, 4)

binary_model.load_weights(r'C:\Users\kesch\OneDrive\Documents\Deeplearning\seg_weights\whole.hdf5')

batch_size = 6

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

train_img_datagen = imageLoader_crop(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
val_img_datagen = imageLoader_val_crop(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size, binary_model)

LR = 0.0003
optim = LH_Adam(LR)

model.compile(optimizer=optim, loss=log_cosh_dice_loss, metrics=[dice_coef_multilabel, core_tumor,
                                                                 peritumoral_edema, enhancing_tumor])

checkpoint_path = r'C:\Users\kesch\OneDrive\Documents\Deeplearning\seg_weights\combined_working.hdf5'

callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_best_only=True)

# model.load_weights(checkpoint_path)
#
# model.fit(train_img_datagen,
#           steps_per_epoch=steps_per_epoch,
#           epochs=300, verbose=1,
#           validation_data=val_img_datagen,
#           validation_steps=val_steps_per_epoch,
#           callbacks=[callback])
