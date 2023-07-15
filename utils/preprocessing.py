from numpy import ndarray
import numpy as np


def calc_z_score(img: ndarray, img_height: int = 128, img_width: int = 128, img_depth: int = 128) -> ndarray:
    avg_pixel_value = np.sum(img) / np.count_nonzero(img)
    sd_pixel_value = np.std(img[np.nonzero(img)])

    for i in range(img_width):
        for j in range(img_height):
            for k in range(img_depth):
                if img[i, j, k] != 0:
                    img[i, j, k] = (img[i, j, k] - avg_pixel_value) / sd_pixel_value

    return img


def normalize_mri_data(t1: ndarray, t1ce: ndarray, t2: ndarray, flair: ndarray, mask: ndarray) \
        -> tuple[ndarray, ndarray]:
    t2 = t2[56:184, 56:184, 13:141].reshape(-1, t2.shape[-1]).reshape(t2.shape)
    t2 = calc_z_score(t2)

    t1ce = t1ce[56:184, 56:184, 13:141].reshape(-1, t1ce.shape[-1]).reshape(t1ce.shape)
    t1ce = calc_z_score(t1ce)

    flair = flair[56:184, 56:184, 13:141].reshape(-1, flair.shape[-1]).reshape(flair.shape)
    flair = calc_z_score(flair)

    t1 = t1[56:184, 56:184, 13:141].reshape(-1, t1.shape[-1]).reshape(t1.shape)
    t1 = calc_z_score(t1)

    mask = mask.astype(np.uint8)
    mask[mask == 4] = 3
    mask = mask[56:184, 56:184, 13:141]

    data = np.stack([flair, t1ce, t1, t2], axis=3)

    return data, mask


def roi_crop(img: ndarray, mask: ndarray, model) -> tuple[ndarray, ndarray]:
    img_input = np.expand_dims(img, axis=0)

    binary_mask = model.predict(img_input)
    binary_mask = binary_mask[0, :, :, :, 0]

    loc = np.where(binary_mask == 1)
    a = max(0, np.amin(loc[0]) - 12)
    b = min(128, np.amax(loc[0]) + 12)
    c = max(0, np.amin(loc[1]) - 12)
    d = min(128, np.amax(loc[1]) + 12)
    e = max(0, np.amin(loc[2]) - 12)
    f = min(128, np.amax(loc[2]) + 12)

    return img[a:b, c:d, e:f], mask[a:b, c:d, e:f]


def mask_to_binary_mask(mask: ndarray) -> ndarray:
    mask = mask.copy()
    for batch in range(mask.shape[0]):
        for height in range(mask.shape[1]):
            for width in range(mask.shape[2]):
                for depth in range(mask.shape[3]):
                    mask[batch][height][width][depth] = 0 if mask[batch][height][width][depth] == [1, 0, 0, 0] else 1
    return mask
