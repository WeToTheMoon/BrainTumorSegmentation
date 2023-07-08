import keras.backend as kb


def dice_coef(y_true, y_pred, smooth=1):
    """
    Takes the TRUE and PRED and returns the dice score
    :param smooth:
    :param y_true:
    :param y_pred:
    """
    y_true_f = kb.flatten(y_true)
    y_pred_f = kb.flatten(y_pred)
    intersection = kb.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (kb.sum(y_true_f) + kb.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred):
    """
    Takes the TRUE and PRED and returns overall dice score across multiple regions
    :param y_true:
    :param y_pred:
    """
    return (core_tumor(y_true, y_pred) + peritumoral_edema(y_true, y_pred) + enhancing_tumor(y_true, y_pred)) / 3


def core_tumor(y_true, y_pred):
    """
    Computes the dice score of the core_tumor regions
    :param y_true:
    :param y_pred:
    """
    return dice_coef(y_true[:, :, :, :, 1], y_pred[:, :, :, :, 1])


def peritumoral_edema(y_true, y_pred):
    """
    Computes the dice score of the peritumoral edema region
    :param y_true:
    :param y_pred:
    """
    return dice_coef(y_true[:, :, :, :, 2], y_pred[:, :, :, :, 2])


def enhancing_tumor(y_true, y_pred):
    """
    Computes the dice core of the enhancing tumor region
    :param y_true:
    :param y_pred:
    """
    return dice_coef(y_true[:, :, :, :, 3], y_pred[:, :, :, :, 3])
