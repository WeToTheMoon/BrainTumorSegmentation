import numpy as np
from augmentations import combine_aug
import random
import os

def load_img(img_dir, img_list):
    '''
    loads the image and the mask
    :param img_dir:
    :param img_list:
    '''
    images = []
    for i, image_name in enumerate(img_list):
        if (image_name.split('.')[1] == 'npy'):
            image = np.load(os.path.join(img_dir, image_name))
            images.append(image)
    return np.array(images)

def load_img_cropped(img_dir, img_list):
    '''
    loads the image and the mask
    :param img_dir:
    :param img_list:
    '''
    images = []
    for i, image_name in enumerate(img_list):
        if (image_name.split('.')[1] == 'npy'):
            if "image" in image_name:
                image = np.load(os.path.join(img_dir, image_name))
                images.append(image[:,:,:,:-1])
            else:
                image = np.load(os.path.join(img_dir, image_name))
                images.append(image)
    return np.array(images)
def global_extraction(img, mask):
    '''
    Crops the image to 48 x 48 x 128 x C
    :param img:
    :param mask:
    '''
    img1 = []
    mask1 = []
    for i in range(len(img)):
        img_temp = img[i]
        mask_temp = mask[i]
        or_r = np.random.randint(0, img_temp.shape[0]-47)
        or_c = np.random.randint(0, img_temp.shape[1]-47)
        img_temp = img_temp[or_r:or_r+48,or_c:or_c+48,:,:]
        mask_temp = mask_temp[or_r:or_r+48,or_c:or_c+48,:,:]
        img1.append(img_temp)
        mask1.append(mask_temp)
    img_final = np.stack((i for i in img1), axis=0)
    mask_final = np.stack((i for i in mask1), axis=0)
    return img_final, mask_final


def imageLoader_crop(img_dir, img_list, mask_dir, mask_list, batch_size1, model):
    '''
    Generator for the images
    :param img_dir:
    :param img_list:
    :param mask_dir:
    :param mask_list:
    :param batch_size1:
    '''
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
    '''
    Generator for the training images
    :param img_dir:
    :param img_list:
    :param mask_dir:
    :param mask_list:
    :param batch_size1:
    '''
    L = len(img_list)
    while True:

        batch_start = 0
        batch_end = batch_size1

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img_cropped(img_dir, img_list[batch_start:limit])
            Y = load_img_cropped(mask_dir, mask_list[batch_start:limit])
            X, Y = global_extraction(X, Y)

            region_based = np.zeros((batch_size1, 48, 48, 128, 1))

            region_based = model.predict(X, verbose = 0)

            X = np.concatenate((X, region_based), axis=-1)

            batch_start += batch_size1
            batch_end += batch_size1
            yield (X, Y)


def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size1):
    '''
    Loads images without cropping
    :param img_dir:
    :param img_list:
    :param mask_dir:
    :param mask_list:
    :param batch_size1:
    '''
    # keras needs the generator infinite, so we will use while true
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

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            combine_aug(X, Y)

            batch_start += batch_size1
            batch_end += batch_size1

            yield (X, Y)  # a tuple with two numpy arrays with batch_size samples


def imageLoader_val(img_dir, img_list, mask_dir, mask_list, batch_size1):
    '''
    Loads val images without cropping
    :param img_dir:
    :param img_list:
    :param mask_dir:
    :param mask_list:
    :param batch_size1:
    '''
    L = len(img_list)

    # keras needs the generator infinite, so we will use while true
    while True:

        batch_start = 0
        batch_end = batch_size1

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X, Y)  # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size1
            batch_end += batch_size1
