import numpy as np
from utils.models import binary_model
from keras.layers import Input, Conv3D, concatenate, Conv3DTranspose, LeakyReLU
from keras.models import Model
from tensorflow_addons.layers import InstanceNormalization


# np.set_printoptions(threshold=sys.maxsize)
# #Group Images
# testing_imgs = glob(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\*.npy")
# training_imgs = glob(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\train\images\*.npy")
# #Group Masks
# mask_imgs = glob(r"C:\Users\kesch\OneDrive\Desktop\BratsSeg1\val\masks\*.npy")

# Create Cropped Photo with 0 pooling
def crop_photo(img_orig):
    # File Path of image, mask of corresponding image, image
    a_list = []
    b_list = []
    c_list = []
    d_list = []

    model = binary_model(128, 128, 128, 4, 1)

    model.load_weights(r'C:\Users\kesch\OneDrive\Documents\Deeplearning\seg_weights\binary_growth.hdf5')
    test_img_input = np.expand_dims(img_orig, axis=0)
    test_prediction = model.predict(test_img_input)
    img_temp = test_prediction[0, :, :, :, 0]
    for i in range(128):
        if np.sum(img_temp[:, :, i]) >= 1:
            location = np.where(img_temp[:, :, i] == 1)
            a = np.amin(location[0])
            b = np.amax(location[0])
            c = np.amin(location[1])
            d = np.amax(location[1])

            a -= 12
            b += 12
            c -= 12
            d += 12

            a = max(a, 0)
            b = min(b, 128)
            c = max(c, 0)
            d = min(d, 128)

            a_list.append(a)
            b_list.append(b)
            c_list.append(c)
            d_list.append(d)

    a = np.amin(a_list)
    b = np.amax(b_list)
    c = np.amin(c_list)
    d = np.amax(d_list)

    print(a)
    print(b)
    print(c)
    print(d)
    image_new = img_orig[a:b, c:d, :]
    image_bin = img_temp[a:b, c:d, :]  # Change to test_prediction

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
