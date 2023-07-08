from loader import imageLoader_crop, imageLoader_val_crop
from models import Multiclass_model, binary_model
import keras
from metric import dice_coef_multilabel, enhancing_tumor, peritumoral_edema, core_tumor
from keras.callbacks import ModelCheckpoint
import os
from loss import log_cosh_dice_loss
from optimizer import LH_Adam
train_img_dir = os.path.realpath(r'/mnt/c/Users/kesch/OneDrive/Desktop/BrainTumorSeg/train/images')
train_mask_dir = os.path.realpath(r'/mnt/c/Users/kesch/OneDrive/Desktop/BrainTumorSeg/train/masks')
val_img_dir = os.path.realpath(r'/mnt/c/Users/kesch/OneDrive/Desktop/BrainTumorSeg/val/images')
val_mask_dir = os.path.realpath(r'/mnt/c/Users/kesch/OneDrive/Desktop/BrainTumorSeg/val/masks')

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

batch_size = 6

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

model_whole = binary_model()
model_whole.load_weights("Segmentation weights/binary_model.hdf5")

train_img_datagen = imageLoader_crop(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size, model_whole)

val_img_datagen = imageLoader_val_crop(val_img_dir, val_img_list,
                                  val_mask_dir, val_mask_list, batch_size, model_whole)

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

LR = 0.0003
optim = LH_Adam()
Metrics = [dice_coef_multilabel, enhancing_tumor, peritumoral_edema, core_tumor]

model = Multiclass_model()
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=Metrics)

checkpoint_path = './Segmentation weights/Unet.hdf5'
callback = ModelCheckpoint(filepath=checkpoint_path,
                           save_weights_only=True, verbose=1)
history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=300,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch,
                    callbacks=[callback],
                    )
