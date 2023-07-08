import os
from argparse import ArgumentParser
from os import path
from glob import glob

from utils.loader import imageLoader_crop, imageLoader_val_crop
from utils.loss import log_cosh_dice_loss
from utils.metrics import dice_coef_multilabel, peritumoral_edema, enhancing_tumor, core_tumor
from utils.models import brain_tumor_model as BrainTumorModel, binary_model as BinaryModel
from utils.optimizers import LH_Adam


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
    arg_parser.add_argument("-w", "--binary_weights", help="Path to the binary model's weights", required=True)

    args = arg_parser.parse_args()

    # Check Weights File
    if not path.isfile(args.binary_weights):
        raise FileExistsError("Unable to find the weights file")

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

    binary_model = BinaryModel(128, 128, 128, 4, 1, 20)
    binary_model.load_weights(args.binary_weights)

    brain_tumor_model = BrainTumorModel(48, 48, 128, 5, 4)

    batch_size = 6

    steps_per_epoch = len(train_img_list) // batch_size
    val_steps_per_epoch = len(val_img_list) // batch_size

    train_img_datagen = imageLoader_crop(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size,
                                         brain_tumor_model)
    val_img_datagen = imageLoader_val_crop(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size,
                                           binary_model)

    LR = 0.0003
    optim = LH_Adam(LR)

    brain_tumor_model.compile(optimizer=optim, loss=log_cosh_dice_loss, metrics=[dice_coef_multilabel, core_tumor,
                                                                                 peritumoral_edema, enhancing_tumor])

    brain_tumor_model.fit(train_img_datagen,
                          steps_per_epoch=steps_per_epoch,
                          epochs=300, verbose=1,
                          validation_data=val_img_datagen,
                          validation_steps=val_steps_per_epoch)


if __name__ == '__main__':
    main()
