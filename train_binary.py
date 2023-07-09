import os
from argparse import ArgumentParser
from glob import glob
from os import path

from keras.callbacks import ModelCheckpoint
from tensorflow import keras

from utils.loader import imageLoader_val, imageLoader
from utils.loss import dice_coef_loss
from utils.metrics import dice_coef
from utils.models import binary_model


def main():
    arg_parser = ArgumentParser()

    arg_parser.add_argument("-t", "--train_dir",
                            help="Directory for the training dataset. Should contain the 'images' and 'masks' "
                                 "directories.",
                            required=True)
    arg_parser.add_argument("-v", "--val_dir",
                            help="Directory for the validation dataset. Should contain the 'images' and 'masks' "
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
    if not path.isdir(args.train_dir):
        raise FileExistsError("Unable to find the training directory")

    if not path.isdir(f"{args.train_dir}/images") or not path.isdir(f"{args.train_dir}/masks"):
        raise FileExistsError("Unable to find the images and/or masks directory within the training directory")

    training_files = glob(f"{args.train_dir}/*/*/*.npy")
    if not all([file.endswith(".npy") for file in training_files]) or len(training_files) == 0:
        raise ValueError("The training directory doesn't consist of all .npy files. Make sure that only .npy files "
                         "are present.")

    # Check Validation Directory
    if not path.isdir(args.val_dir):
        raise FileExistsError("Unable to find the validation directory")

    if not path.isdir(f"{args.train_dir}/images") or not path.isdir(f"{args.train_dir}/masks"):
        raise FileExistsError("Unable to find the images and/or masks directory within the validation directory")

    validation_files = glob(f"{args.val_dir}/*/*/*.npy")
    if not all([file.endswith(".npy") for file in validation_files]) or len(validation_files) == 0:
        raise ValueError("The training directory doesn't consist of all .npy files. Make sure that only .npy files "
                         "are present.")

    train_img_dir = f"{args.train_dir}/images"
    train_mask_dir = f"{args.train_dir}/masks"

    val_img_dir = f"{args.val_dir}/images"
    val_mask_dir = f"{args.val_dir}/masks"

    train_img_list = os.listdir(train_img_dir)
    train_mask_list = os.listdir(train_mask_dir)

    val_img_list = os.listdir(val_img_dir)
    val_mask_list = os.listdir(val_mask_dir)

    b_size = 2

    train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                    train_mask_dir, train_mask_list, b_size)
    val_img_datagen = imageLoader_val(val_img_dir, val_img_list,
                                      val_mask_dir, val_mask_list, b_size)

    LR = 0.0003
    optim = keras.optimizers.Adam(LR)

    steps_per_epoch = len(train_img_list) // b_size
    val_steps_per_epoch = len(val_img_list) // b_size

    n_channels = 20
    model = binary_model(128, 128, 128, 4, 1, n_channels)

    model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])

    callback = ModelCheckpoint(filepath=args.binary_weights, save_weights_only=True)

    model.fit(train_img_datagen,
              steps_per_epoch=steps_per_epoch,
              epochs=1000,
              verbose=1,
              validation_data=val_img_datagen,
              validation_steps=val_steps_per_epoch,
              callbacks=[callback])


if __name__ == '__main__':
    main()
