from numpy import ndarray
import numpy as np


def calc_z_score(img: ndarray, img_height: int, img_width: int, img_depth: int) -> ndarray:
    avg_pixel_value = np.sum(img) / np.count_nonzero(img)
    sd_pixel_value = np.std(img[np.nonzero(img)])

    for i in range(img_width):
        for j in range(img_height):
            for k in range(img_depth):
                if img[i, j, k] != 0:
                    img[i, j, k] = (img[i, j, k] - avg_pixel_value) / sd_pixel_value

    return img

