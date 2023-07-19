from argparse import ArgumentParser
from keras.callbacks import ModelCheckpoint
from keras.layers import ELU
from utils.dataset import MRIDataset
from utils.loss import dice_loss_binary
from utils.metric import dice_coef
from utils.models import binary_model
from utils.optimizers import LH_Adam


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

    dataset = MRIDataset(binary_dataset_path=args.dataset_dir)

    batch_size = 2
    train_img_datagen = dataset.binary_train_datagen(batch_size)
    val_img_datagen = dataset.binary_val_datagen(batch_size)

    steps_per_epoch, val_steps_per_epoch = dataset.binary_steps_per_epoch(batch_size)

    n_channels = 20
    model = binary_model(128, 128, 128, 4, 1, n_channels, activation=ELU())

    learning_rate = 0.0003
    optimizer = LH_Adam(learning_rate)

    model.compile(optimizer=optimizer, loss=dice_loss_binary, metrics=[dice_coef])

    callback = ModelCheckpoint(filepath=args.binary_weights, save_weights_only=True, save_best_only=True)

    model.fit(x=train_img_datagen,
              steps_per_epoch=steps_per_epoch,
              epochs=200,
              verbose=1,
              validation_data=val_img_datagen,
              validation_steps=val_steps_per_epoch,
              callbacks=[callback],
              # TODO test that this works. This should function to increase the training speed but might not work if because we infinitely return batches from our datagen
              use_multiprocessing=True)


if __name__ == '__main__':
    main()
