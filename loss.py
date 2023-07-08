from metric import dice_coef_multilabel, peritumoral_edema, core_tumor, enhancing_tumor
import tensorflow as tf

def CE(y_true, y_pred):
    return tf.keras.losses.CategoricalCrossentropy(y_true, y_pred)

def dice_loss(y_true, y_pred):
    '''
    Takes the normal dice loss (Not weighted)
    :param y_true:
    :param y_pred:
    '''
    return (1-dice_coef_multilabel(y_true, y_pred))



