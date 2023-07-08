from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
from keras.optimizers import Adam


def LH_Adam(learning_rate: float):
    return Lookahead(Adam(learning_rate))


def Ranger(learning_rate: float):
    return Lookahead(RectifiedAdam(learning_rate))
