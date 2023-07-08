import numpy as np
from utils.augmentations import combine_aug
import random
import os


def load_img(img_dir: str, img_list):
    """
    loads the image and the mask
    :param img_dir:
    :param img_list:
    """
    images = []
    for i, image_name in enumerate(img_list):
        if image_name.split('.')[1] == 'npy':
            image = np.load(os.path.join(img_dir, image_name))
            images.append(image)
    return np.array(images)


def load_img_cropped(img_dir, img_list):
    """
    loads the image and the mask
    :param img_dir:
    :param img_list:
    """

    # Does not load the binary mask
    images = []
    for i, image_name in enumerate(img_list):
        if image_name.split('.')[1] == 'npy':
            if "image" in image_name:
                image = np.load(os.path.join(img_dir, image_name))
                images.append(image[:, :, :, :-1])
            else:
                image = np.load(os.path.join(img_dir, image_name))
                images.append(image)
    return np.array(images)


def global_extraction(img, mask):
    """
    Crops the image to 48 x 48 x 128 x C
    :param img:
    :param mask:
    """
    images = []
    masks = []
    for i in range(len(img)):
        img_temp = img[i]
        mask_temp = mask[i]
        or_r = np.random.randint(0, img_temp.shape[0] - 47)
        or_c = np.random.randint(0, img_temp.shape[1] - 47)
        img_temp = img_temp[or_r:or_r + 48, or_c:or_c + 48, :, :]
        mask_temp = mask_temp[or_r:or_r + 48, or_c:or_c + 48, :, :]
        images.append(img_temp)
        masks.append(mask_temp)

    stacked_images = np.stack(images, axis=0)
    stacked_masks = np.stack(masks, axis=0)

    return stacked_images, stacked_masks


def imageLoader_crop(img_dir, img_list, mask_dir, mask_list, batch_size1, model):
    """
    Generator for the images when predicting the multiclass mask in the train set
    :param img_dir:
    :param img_list:
    :param mask_dir:
    :param mask_list:
    :param batch_size1:
    :param model:
    """
    while True:
        batch_start = 0
        batch_end = batch_size1

        temp1 = list(zip(img_list, mask_list))
        random.shuffle(temp1)
        img_list, mask_list = zip(*temp1)
        img_list, mask_list = list(img_list), list(mask_list)

        L = len(img_list)

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img_cropped(img_dir, img_list[batch_start:limit])
            Y = load_img_cropped(mask_dir, mask_list[batch_start:limit])
            X, Y = global_extraction(X, Y)

            region_based = model.predict(X, verbose=0)

            X = np.concatenate((X, region_based), axis=-1)

            combine_aug(X, Y)
            batch_start += batch_size1
            batch_end += batch_size1

            yield X, Y


def imageLoader_val_crop(img_dir, img_list, mask_dir, mask_list, batch_size1, model):
    """
    Generator for the images when predicting the multiclass mask in the validation set
    :param img_dir:
    :param img_list:
    :param mask_dir:
    :param mask_list:
    :param batch_size1:
    :param model:
    """
    L = len(img_list)
    while True:
        batch_start = 0
        batch_end = batch_size1

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img_cropped(img_dir, img_list[batch_start:limit])
            Y = load_img_cropped(mask_dir, mask_list[batch_start:limit])
            X, Y = global_extraction(X, Y)

            region_based = model.predict(X, verbose=0)

            X = np.concatenate((X, region_based), axis=-1)

            batch_start += batch_size1
            batch_end += batch_size1
            yield X, Y


def imageLoader(img_dir: str, img_list, mask_dir: str, mask_list, batch_size: int):
    """
    Loads images without cropping
    :param img_dir:
    :param img_list:
    :param mask_dir:
    :param mask_list:
    :param batch_size:
    """
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

            combine_aug(X, Y)

            batch_start += batch_size
            batch_end += batch_size

            yield X, Y  # a tuple with two numpy arrays with batch_size samples


def imageLoader_val(img_dir, img_list, mask_dir, mask_list, batch_size1):
    """
    Loads val images without cropping
    :param img_dir:
    :param img_list:
    :param mask_dir:
    :param mask_list:
    :param batch_size1:
    """
    L = len(img_list)

    # keras needs the generator infinite, so we will use while true
    while True:
        batch_start = 0
        batch_end = batch_size1

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            batch_start += batch_size1
            batch_end += batch_size1

            yield X, Y  # a tuple with two numpy arrays with batch_size samples