from keras.layers import Conv3D, Conv3DTranspose, Input, concatenate, BatchNormalization, LeakyReLU, MaxPool3D
from keras.models import Model
from tensorflow_addons.layers import InstanceNormalization


def double_conv_block1(x, n_filters, activation):
    x1 = InstanceNormalization()(x)
    x1 = Conv3D(n_filters, 3, padding="same", activation=activation)(x1)
    x2 = InstanceNormalization()(x1)
    x2 = Conv3D(n_filters, 3, padding="same", activation=activation)(x2)
    return x2


def binary_model():
    """
    Constructor for the bianry model
    :return:
    """
    leaky_relu = LeakyReLU(alpha=0.05)
    channels = 32
    inputs = Input((48, 48, 128, 4))

    c1 = double_conv_block(inputs, channels * 2, leaky_relu)
    p1 = Conv3D(channels * 2, (3, 3, 3), strides=(2, 2, 2), activation=leaky_relu, padding='same')(c1)

    c2 = double_conv_block(p1, channels * 4, leaky_relu)
    p2 = Conv3D(channels * 4, (3, 3, 3), strides=(2, 2, 2), activation=leaky_relu, padding='same')(c2)

    c3 = double_conv_block(p2, channels * 8, leaky_relu)
    p3 = Conv3D(channels * 8, (3, 3, 3), strides=(2, 2, 2), activation=leaky_relu, padding='same')(c3)

    c4 = double_conv_block(p3, channels * 10, leaky_relu)

    u1 = Conv3DTranspose(channels * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c4)
    u1 = concatenate([u1, c3])
    c5 = double_conv_block(u1, channels * 8, leaky_relu)

    u2 = Conv3DTranspose(channels * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u2 = concatenate([u2, c2])
    c6 = double_conv_block(u2, channels * 4, leaky_relu)

    u3 = Conv3DTranspose(channels * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u3 = concatenate([u3, c1])
    c7 = double_conv_block(u3, channels * 2, leaky_relu)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def double_conv_block(x, n_filters, activation):
    """
    Convolution block used in the encoder and decoder
    :param x:
    :param n_filters:
    :param activation:
    :return:
    """
    x1 = BatchNormalization()(x)
    x1 = BatchNormalization(axis=4)(x1)
    x1 = Conv3D(n_filters, 3, padding="same", activation=activation)(x1)
    x2 = BatchNormalization()(x1)
    x2 = BatchNormalization(axis=4)(x2)
    x2 = Conv3D(n_filters, 3, padding="same", activation=activation)(x2)
    return x2


def multiclass_model(img_height: int, img_width: int,
                     img_depth: int, img_channels: int,
                     num_classes: int, activation=LeakyReLU(alpha=0.05), channels: int = 32):
    inputs = Input((img_height, img_width, img_depth, img_channels))

    if num_classes > 1:
        function = 'softmax'
    else:
        function = 'sigmoid'

    c1 = double_conv_block1(inputs, channels * 2, activation)
    p1 = Conv3D(channels * 2, (3, 3, 3), strides=(2, 2, 2), activation=activation, padding='same')(c1)

    c2 = double_conv_block1(p1, channels * 4, activation)
    p2 = Conv3D(channels * 4, (3, 3, 3), strides=(2, 2, 2), activation=activation, padding='same')(c2)

    c3 = double_conv_block1(p2, channels * 8, activation)
    p3 = Conv3D(channels * 8, (3, 3, 3), strides=(2, 2, 2), activation=activation, padding='same')(c3)

    c4 = double_conv_block1(p3, channels * 10, activation)

    u1 = Conv3DTranspose(channels * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c4)
    u1 = concatenate([u1, c3])
    c5 = double_conv_block1(u1, channels * 8, activation)

    u2 = Conv3DTranspose(channels * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u2 = concatenate([u2, c2])
    c6 = double_conv_block1(u2, channels * 4, activation)

    u3 = Conv3DTranspose(channels * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u3 = concatenate([u3, c1])
    c7 = double_conv_block1(u3, channels * 2, activation)

    outputs = Conv3D(num_classes, (1, 1, 1), activation=function)(c7)

    return Model(inputs=[inputs], outputs=[outputs])


def double_conv_block_Unet(x, n_filters):
    x1 = Conv3D(n_filters, 3, padding="same", activation='relu')(x)
    x2 = Conv3D(n_filters, 3, padding="same", activation='relu')(x1)
    return x2


def Unet_model():
    """
    Unet architecture
    :return:
    """
    channels = 16
    inputs = Input((128, 128, 128, 4))

    c1 = double_conv_block_Unet(inputs, channels * 2)
    p1 = MaxPool3D((2, 2, 2))(c1)

    c2 = double_conv_block_Unet(p1, channels * 4)
    p2 = MaxPool3D((2, 2, 2))(c2)

    c3 = double_conv_block_Unet(p2, channels * 8)
    p3 = MaxPool3D((2, 2, 2))(c3)

    c4 = double_conv_block_Unet(p3, channels * 16)
    p4 = MaxPool3D((2, 2, 2))(c4)

    c5 = double_conv_block_Unet(p4, channels * 32)

    u0 = Conv3DTranspose(channels * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u0 = concatenate([u0, c4])
    c6 = double_conv_block_Unet(u0, channels * 16)

    u1 = Conv3DTranspose(channels * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u1 = concatenate([u1, c3])
    c7 = double_conv_block_Unet(u1, channels * 8)

    u2 = Conv3DTranspose(channels * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u2 = concatenate([u2, c2])
    c8 = double_conv_block_Unet(u2, channels * 4)

    u3 = Conv3DTranspose(channels * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u3 = concatenate([u3, c1])
    c9 = double_conv_block_Unet(u3, channels * 2)

    outputs = Conv3D(4, (1, 1, 1), activation='softmax')(c9)

    return Model(inputs=[inputs], outputs=[outputs])