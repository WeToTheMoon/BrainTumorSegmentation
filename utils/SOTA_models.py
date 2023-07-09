from keras.layers import Conv3D, Conv3DTranspose, Input, concatenate, MaxPool3D
from keras.models import Model


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
