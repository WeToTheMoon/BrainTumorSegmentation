import numpy as np
import elasticdeform

def brightness(X, y):
    '''
    Applies brightness to X and Y
    :param X:
    :param y:
    '''
    X_new = np.zeros(X.shape)
    for c in range(X.shape[-1]):
        im = X[:, :, :, c]
        gain = np.random.uniform(0.8,2.3)
        gamma = np.random.uniform(0.8,2.3)
        im_new = np.sign(im)*gain*(np.abs(im)**gamma)
        X_new[:, :, :, c] = im_new

    return X_new, y

def elastic(X, y, sigma):
    '''
    Applies elastic deformation to the X and Y equally. The severity is determined by the value of sigma
    :param X:
    :param y:
    :param sigma:
    '''
    [Xel, yel] = elasticdeform.deform_random_grid(
        [X, y], sigma=np.random.uniform(sigma[0], sigma[1]), axis=[(0, 1, 2), (0, 1, 2),], order=[1, 0], mode='constant')

    return Xel, yel


def combine_aug(x, y):
    '''
    Combines the brighness and elastic deformation augmentations with a 30% of each augmentation being applied
    :param x:
    :param y:
    '''
    Xnew, ynew = x, y

    prob = np.random.randint(0, 10)
    prob1 = np.random.randint(0, 10)

    if prob < 3:
        Xnew, ynew = brightness(Xnew, ynew)

    if prob1 < 3:
        Xnew, ynew = elastic(Xnew, ynew, (10., 13.))
    return Xnew, ynew

