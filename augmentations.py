import numpy as np
import elasticdeform


def brightness(X, y):
    """
    Applies brightness to X and Y
    :param X:
    :param y:
    """
    x_new = np.zeros(X.shape)
    for c in range(X.shape[-1]):
        im = X[:, :, :, c]
        gain = np.random.uniform(0.8, 2.3)
        gamma = np.random.uniform(0.8, 2.3)
        im_new = np.sign(im) * gain * (np.abs(im) ** gamma)
        x_new[:, :, :, c] = im_new

    return x_new, y


def elastic(x, y, sigma):
    """
    Applies elastic deformation to the X and Y equally. The severity is determined by the value of sigma
    :param x:
    :param y:
    :param sigma:
    """
    [x_el, y_el] = elasticdeform.deform_random_grid(
        [x, y], sigma=np.random.uniform(sigma[0], sigma[1]), axis=[(0, 1, 2), (0, 1, 2)], order=[1, 0],
        mode='constant')

    return x_el, y_el


def combine_aug(x, y):
    """
    Combines the brightness and elastic deformation augmentations with a 30% of each augmentation being applied
    :param x:
    :param y:
    """
    x_new, y_new = x, y

    prob = np.random.randint(0, 10)
    prob1 = np.random.randint(0, 10)

    if prob < 3:
        x_new, y_new = brightness(x_new, y_new)

    if prob1 < 3:
        x_new, y_new = elastic(x_new, y_new, (10., 13.))
    return x_new, y_new
