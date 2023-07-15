import os
from argparse import ArgumentParser
from os import path
from glob import glob

from utils.loader import cropped_image_loader, cropped_image_loader_val
from utils.loss import log_cosh_dice_loss, log_cosh_dice_loss_binary
from utils.metric import dice_coef_multilabel, peritumoral_edema, enhancing_tumor, core_tumor, dice_coef
from utils.models import attention_brain_tumor_model as BrainTumorAttentionModel
from utils.optimizers import LH_Adam
from keras.callbacks import ModelCheckpoint

def main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("-d", "--dataset_dir",
                            help="Directory for the training dataset. Should contain the 'train' and 'val' "
                                 "directories.",
                            required=True)
    arg_parser.add_argument("-w", "--binary_weights",
                            help="Path to the binary model's weights. This is the path of where to save the weights",
                            required=True)

    args = arg_parser.parse_args()

    # Check Weights File
    if not args.binary_weights.endswith(".hdf5"):
        raise ValueError("Invalid weight file format")

    # Check Training Directory
    if not path.isdir(args.dataset_dir):
        raise FileExistsError("Unable to find the dataset directory")

    if not path.isdir(f"{args.dataset_dir}/train") or not path.isdir(f"{args.dataset_dir}/val"):
        raise FileExistsError("Unable to find the images and/or masks directory within the training directory")

    training_files = glob(f"{args.dataset_dir}/*/*/*.npy")
    print(len(training_files))
    if not all([file.endswith(".npy") for file in training_files]) or len(training_files) == 0:
        raise ValueError("The training directory doesn't consist of all .npy files. Make sure that only .npy files "
                         "are present.")

    train_img_dir = f"{args.dataset_dir}/train/images"
    train_mask_dir = f"{args.dataset_dir}/train/masks"

    val_img_dir = f"{args.dataset_dir}/val/images"
    val_mask_dir = f"{args.dataset_dir}/val/masks"

    train_img_list = os.listdir(train_img_dir)
    train_mask_list = os.listdir(train_mask_dir)

    val_img_list = os.listdir(val_img_dir)
    val_mask_list = os.listdir(val_mask_dir)

    batch_size = 6

    steps_per_epoch = len(train_img_list) // batch_size
    val_steps_per_epoch = len(val_img_list) // batch_size

    train_img_datagen = cropped_image_loader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)

    val_img_datagen = cropped_image_loader_val(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)

    brain_tumor_model = BrainTumorAttentionModel(48, 48, 48, 4, 4)

    learning_rate = 0.0003
    optim = LH_Adam(learning_rate)

    brain_tumor_model.compile(optimizer=optim, loss=log_cosh_dice_loss_binary, metrics=[dice_coef])

    callback = ModelCheckpoint(filepath=args.binary_weights, save_weights_only=True, save_best_only=True)

    history = brain_tumor_model.fit(train_img_datagen,
                              steps_per_epoch=steps_per_epoch,
                              epochs=500, verbose=1,
                              validation_data=val_img_datagen,
                              validation_steps=val_steps_per_epoch)

    print(max(history.history["val_dice_coef_multilabel"]))

    # Brain tumor attention model with self attention


if __name__ == '__main__':
    main()
