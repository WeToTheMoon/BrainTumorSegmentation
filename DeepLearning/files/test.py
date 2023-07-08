import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import sys
from keras.layers import Input, Conv3D, concatenate, Conv3DTranspose, LeakyReLU
from tensorflow_addons.layers import InstanceNormalization
from keras.models import Model
import tensorflow as tf
# np.set_printoptions(threshold=sys.maxsize)
# #Group Images
# testing_imgs = glob(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\*.npy")
# training_imgs = glob(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\train\images\*.npy")
# #Group Masks
# mask_imgs = glob(r"C:\Users\kesch\OneDrive\Desktop\BratsSeg1\val\masks\*.npy")

#Conv layers of binary masks
def double_conv_block(x, n_filters):
    leaky_relu = LeakyReLU(alpha=0.01)
    x = InstanceNormalization()(x)
    x = Conv3D(n_filters, 3, padding="same", activation=leaky_relu)(x)
    x1 = InstanceNormalization()(x)
    x1 = Conv3D(n_filters, 3, padding="same", activation=leaky_relu)(x)
    x1 = concatenate([x1, x])
    x2 = InstanceNormalization()(x1)
    x2 = Conv3D(n_filters, 3, padding="same", activation=leaky_relu)(x2)
    x2 = InstanceNormalization()(x1)
    x2 = Conv3D(n_filters, 3, padding="same", activation=leaky_relu)(x2)
    return x2

#Model
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
#Build the model
    leaky_relu = LeakyReLU(alpha=0.01)
    n_channels = 16
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = double_conv_block(s, n_channels)
    p1 = Conv3D(n_channels, (3, 3, 3), strides = (2,2,2), activation=leaky_relu, padding='same')(c1)

    c2 = double_conv_block(p1, n_channels*2)
    p2 = Conv3D(n_channels*2, (3, 3, 3), strides = (2,2,2), activation=leaky_relu, padding='same')(c2)

    c3 = double_conv_block(p2, n_channels*4)
    p3 = Conv3D(n_channels*4, (3, 3, 3), strides = (2,2,2), activation=leaky_relu, padding='same')(c3)

    c4 = double_conv_block(p3, n_channels*8)
    p4 = Conv3D(n_channels*8, (3, 3, 3), strides = (2,2,2), activation=leaky_relu, padding='same')(c4)

    c5 = double_conv_block(p4, n_channels*16)

    #Expansive path
    u6 = Conv3DTranspose(n_channels*8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = double_conv_block(u6, n_channels*8)

    u7 = Conv3DTranspose(n_channels*4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = double_conv_block(u7, n_channels*4)

    u8 = Conv3DTranspose(n_channels*2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = double_conv_block(u8, n_channels*2)

    u9 = Conv3DTranspose(n_channels, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = double_conv_block(u9, n_channels*2)

    outputs = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(c9)
        #Try using sigmoid (research papers)
    model = Model(inputs=[inputs], outputs=[outputs])
    #Changed dropout https://github.com/bnsreenu/python_for_microscopists/blob/master/231_234_BraTa2020_Unet_segmentation/simple_3d_unet.py
    #compile model outside of this function to make it flexible.
    return model

#Create Cropped Photo with 0 pooling
def crop_photo(img_orig):
    #File Path of image, mask of corresponding image, image
    a_list = []
    b_list = []
    c_list = []
    d_list = []

    model = simple_unet_model(128, 128, 128, 4, 1)

    model.load_weights(r'C:\Users\kesch\OneDrive\Documents\Deeplearning\seg_weights\binary_growth.hdf5')
    test_img_input = np.expand_dims(img_orig, axis=0)
    test_prediction = model.predict(test_img_input)
    img_temp = test_prediction[0,:,:,:,0]
    for i in range(128):
        if np.sum(img_temp[:,:,i]) >= 1:
            try:
                location = np.where(img_temp[:,:,i] == 1)
                a = np.amin(location[0])
                b = np.amax(location[0])
                c = np.amin(location[1])
                d = np.amax(location[1])
                
                a -= 12
                b += 12
                c -= 12
                d += 12
                
                if a < 0:
                    a = 0
                else:
                    pass
                if b > 128:
                    b = 128
                else:
                    pass
                if c < 0:
                    c = 0
                else:
                    pass
                if d > 128:
                    d = 128
                else:
                    pass
                a_list.append(a)
                b_list.append(b)
                c_list.append(c)
                d_list.append(d)
            except:
                pass
        else:
            pass
    a = np.amin(a_list)
    b = np.amax(b_list)
    c = np.amin(c_list)
    d = np.amax(d_list)
    print(a)
    print(b)
    print(c)
    print(d)
    image_new = img_orig[a:b,c:d,:]
    image_bin = img_temp[a:b,c:d,:] # Change to test_prediction
    
    # image_fin = np.concatenate((image_new, image_bin), axis = -1)
    return img_temp, image_bin

img = np.load(r"C:\Users\kesch\OneDrive\Desktop\Bratsimages_normalized_onlybrain\images\image_120.npy")
msk = np.load(r"C:\Users\kesch\OneDrive\Desktop\Bratsimages_normalized_onlybrain\masks\mask_120.npy")
img_before_crop, img_crop = crop_photo(img)

# for i in range(128):
#     n_slice = i
#     plt.figure(figsize=(12, 8))
#     plt.subplot(231)
#     plt.title('Before Crop')
#     plt.imshow(img_before_crop[:, :, n_slice], cmap = 'gray')
#     plt.subplot(232)
#     plt.title('After Crop')
#     plt.imshow(img_crop[:, :, n_slice], cmap = 'gray')
#     plt.show()





# for i in glob(r'C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\image*'):
#     loc = i.replace("BrainTumorSeg", "BratsSegBinary")
#     try:
#         img = np.load(loc)
#     except:
#         loc = loc.replace("val", "train")
#         img = np.load(loc)
#     image = crop_photo(img)
#     np.save(i, image)

# for i in glob(r'C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\train\images\image*'):
#     loc = i.replace("BrainTumorSeg", "BratsSegBinary")
#     try:
#         img = np.load(loc)
#     except:
#         loc = loc.replace("train", "val")
#         img = np.load(loc)
#     image = crop_photo(img)
#     np.save(i, image)
    
# img = np.load(R"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\val\images\image_0.npy")
# print(img.shape)
# for i in range(128):
#     plt.figure(figsize=(12, 8))
# #     # plt.subplot(221)
# #     # plt.title('IMG_Binary')
# #     # plt.imshow(T2B_start[:,:,i], cmap='gray')
# #     # plt.subplot(222)
# #     # plt.title('Flair')
# #     # plt.imshow(img[:,:,i,0], cmap='gray')
# #     # plt.subplot(223)
# #     # plt.title('IMG_Binary')
# #     # plt.imshow(T1B[:,:,i], cmap='gray')
# #     # plt.subplot(224)
# #     # plt.title('T1')
# #     # plt.imshow(img[:,:,i,1], cmap='gray')
#     # plt.subplot(232)
#     # plt.title('Original')
#     # plt.imshow(img[:,:,i,0], cmap='gray')
#     # plt.subplot(233)
#     # plt.title('Cropped Mask')
#     # plt.imshow(mask1[:,:,i])
#     # plt.subplot(234)
#     plt.title('img')
#     plt.imshow(img[:,:,i])
#     plt.show()

# #Example
# # for i in training_imgs:
# #     mask_file = i.replace("images\image", "masks\mask")
# #     img = np.load(i)
# #     mask_start = np.load(mask_file)
# #     img, mask = crop_photo(img, mask_start)
# #     img = np.array(img)
# #     mask = np.array(mask)
# #     i = i.replace("BratsSeg1", "BrainTumorSeg")
# #     mask_file = mask_file.replace("BratsSeg1", "BrainTumorSeg")
# #     np.save(i, img)
# #     np.save(mask_file, mask)

# # for i in testing_imgs:
# #     mask_file = i.replace("images\image", "masks\mask")
# #     img = np.load(i)
# #     mask_start = np.load(mask_file)
# #     img, mask = crop_photo(img, mask_start)
# #     img = np.array(img)
# #     mask = np.array(mask)
# #     i = i.replace("BratsSeg1", "BrainTumorSeg")
# #     mask_file = mask_file.replace("BratsSeg1", "BrainTumorSeg")
# #     np.save(i, img)
# #     np.save(mask_file, mask)

# # for i in testing_imgs:
# #     img = np.load(i)
# #     print(i)
# #     print(img.shape)
# # for i in training_imgs:
# #     img = np.load(i)
# #     print(i)
# #     print(img.shape)
# # Index zero is the background
# # Index one is Core Tumor 
# # Index two is peritumoral edema
# # Index three is GD-enhancing tumor