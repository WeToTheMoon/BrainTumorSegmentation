import numpy as np
from numpy import ndarray
from utils.augmentations import combine_aug, binary_combine_aug
import random
import os


def load_img(img_dir: str, img_list: list[str]) -> ndarray:
    """
    TODO add docs

    :param img_dir:
    :param img_list:
    """
    images = []
    for image_name in img_list:
        if image_name.split('.')[1] == 'npy':
            image = np.load(os.path.join(img_dir, image_name))
            images.append(image)
    return np.array(images)


def load_img_cropped(img_dir: str, img_list: list[str]) -> list[ndarray]:
    """
    TODO add docs

    :param img_dir:
    :param img_list:
    """
    # Does not load the binary mask
    images = []
    for image_name in img_list:
        if image_name.split('.')[1] == 'npy':
            image = np.load(os.path.join(img_dir, image_name))
            if "image" in image_name:
                images.append(image[..., :-1])
            else:
                images.append(image)
    return images


def global_extraction(img: ndarray | list[ndarray], mask: ndarray | list[ndarray]) -> tuple[ndarray, ndarray]:
    """
    Crops the image to 48 x 48 x 128 x C.

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
        or_d = np.random.randint(0, img_temp.shape[2] - 47)
        img_temp = img_temp[or_r:or_r + 48, or_c:or_c + 48, or_d:or_d + 48, :]
        mask_temp = mask_temp[or_r:or_r + 48, or_c:or_c + 48, or_d:or_d + 48, :]
        images.append(img_temp)
        masks.append(mask_temp)



    stacked_images = np.stack(images, axis=0)
    stacked_masks = np.stack(masks, axis=0)
    return stacked_images, stacked_masks


def cropped_image_loader(img_dir: str, img_list: list[str],
                         mask_dir: str, mask_list: list[dir],
                         batch_size: int) -> tuple[ndarray, ndarray]:
    """
    Generator for the images when predicting the multiclass mask in the train set.

    :param img_dir:
    :param img_list:
    :param mask_dir:
    :param mask_list:
    :param batch_size:
    :param model:
    """
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

            X = load_img_cropped(img_dir, img_list[batch_start:limit])
            Y = load_img_cropped(mask_dir, mask_list[batch_start:limit])

            X, Y = global_extraction(X, Y)

            print(Y.shape)

            # region_based = model.predict(x, verbose=0)
            #
            # x = np.concatenate((x, region_based), axis=-1)

            combine_aug(X, Y)
            batch_start += batch_size
            batch_end += batch_size

            yield X, Y


def cropped_image_loader_val(img_dir: str, img_list: list[str],
                             mask_dir: str, mask_list: list[dir],
                             batch_size: int) -> tuple[ndarray, ndarray]:
    """
    Generator for the images when predicting the multiclass mask in the validation set.

    :param img_dir:
    :param img_list:
    :param mask_dir:
    :param mask_list:
    :param batch_size:
    :param model:
    """
    L = len(img_list)
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img_cropped(img_dir, img_list[batch_start:limit])
            Y = load_img_cropped(mask_dir, mask_list[batch_start:limit])

            X, Y = global_extraction(X, Y)

            # region_based = model.predict(X, verbose=0)
            #
            # X = np.concatenate((X, region_based), axis=-1)

            batch_start += batch_size
            batch_end += batch_size
            yield X, Y


def image_loader(img_dir: str, img_list: list[str],
                 mask_dir: str, mask_list: list[dir],
                 batch_size: int) -> tuple[ndarray, ndarray]:
    """
    TODO add docs

    :param img_dir:
    :param img_list:
    :param mask_dir:
    :param mask_list:
    :param batch_size:
    """
    # keras needs the generator infinite, so we will use while true
    while True:
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

                x = load_img(img_dir, img_list[batch_start:limit])
                y = load_img(mask_dir, mask_list[batch_start:limit])

                binary_combine_aug(x, y)

                batch_start += batch_size
                batch_end += batch_size

                yield x, y  # a tuple with two numpy arrays with batch_size samples


def image_loader_val(img_dir: str, img_list: list[str],
                     mask_dir: str, mask_list: list[dir],
                     batch_size: int) -> tuple[ndarray, ndarray]:
    """
    TODO add docs

    :param img_dir:
    :param img_list:
    :param mask_dir:
    :param mask_list:
    :param batch_size:
    """
    L = len(img_list)

    # keras needs the generator infinite, so we will use while true
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            batch_start += batch_size
            batch_end += batch_size

            yield X, Y  # a tuple with two numpy arrays with batch_size samples
