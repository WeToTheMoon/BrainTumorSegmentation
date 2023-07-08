from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
from keras.optimizers import Adam


def LH_Adam():
    return Lookahead(Adam())


def Ranger():
    return Lookahead(RectifiedAdam())
