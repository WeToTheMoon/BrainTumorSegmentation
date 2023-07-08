# import numpy as np
# import nibabel as nib
# import glob
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# import os
# from keras.utils.np_utils import to_categorical
# from scipy import stats
# import cv2
# import skimage.color
# scaler = StandardScaler()

# def z_score(img):
#     avg = (np.sum(img))/(np.count_nonzero(img))
#     sd = np.std(img[np.nonzero(img)])
#     for i in range(128):
#         for j in range(128):
#             for k in range(128):
#                 if img[k,j,i]  == 0:
#                     pass
#                 else:
#                     img[k,j,i] = (img[k,j,i] - avg)/sd
#     return img

# t1_list = sorted(glob.glob(r'C:\Users\kesch\OneDrive\Desktop\segmen\*\*t1.nii'))
# t2_list = sorted(glob.glob(r'C:\Users\kesch\OneDrive\Desktop\segmen\*\*t2.nii'))
# t1ce_list = sorted(glob.glob(r'C:\Users\kesch\OneDrive\Desktop\segmen\*\*t1ce.nii'))
# flair_list = sorted(glob.glob(r'C:\Users\kesch\OneDrive\Desktop\segmen\*\*flair.nii'))
# mask_list = sorted(glob.glob(r'C:\Users\kesch\OneDrive\Desktop\segmen\*\*seg.nii'))

# for img in range(len(t2_list)):   #Using t1_list as all lists are of same size
#     print("Now preparing image and masks number: ", img)
      
#     temp_image_t2=nib.load(t2_list[img]).get_fdata()
#     temp_image_t2 = temp_image_t2[56:184, 56:184, 13:141]
#     temp_image_t2=temp_image_t2.reshape(-1, temp_image_t2.shape[-1]).reshape(temp_image_t2.shape)
#     temp_image_t2 = z_score(temp_image_t2)
    
#     temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
#     temp_image_t1ce = temp_image_t1ce[56:184, 56:184, 13:141]
#     temp_image_t1ce=temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1]).reshape(temp_image_t1ce.shape)
#     temp_image_t1ce = z_score(temp_image_t1ce)
   
#     temp_image_flair=nib.load(flair_list[img]).get_fdata()
#     temp_image_flair = temp_image_flair[56:184, 56:184, 13:141]
#     temp_image_flair=temp_image_flair.reshape(-1, temp_image_flair.shape[-1]).reshape(temp_image_flair.shape)
#     temp_image_flair = z_score(temp_image_flair)
    
#     temp_image_t1=nib.load(t1_list[img]).get_fdata()
#     temp_image_t1 = temp_image_t1[56:184, 56:184, 13:141]
#     temp_image_t1=temp_image_t1.reshape(-1, temp_image_t1.shape[-1]).reshape(temp_image_t1.shape)
#     temp_image_t1 = z_score(temp_image_t1)
        
#     temp_mask=nib.load(mask_list[img]).get_fdata()
#     temp_mask=temp_mask.astype(np.uint8)
#     temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
#     temp_mask = temp_mask[56:184, 56:184, 13:141]
#     print(np.unique(temp_mask))
    
#     temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t1, temp_image_t2], axis=3)

    
#     val, counts = np.unique(temp_mask, return_counts=True)
    
#     if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
#         print("Save Me")
#         temp_mask= to_categorical(temp_mask, num_classes=4)
#         np.save(r'C:\Users\kesch\OneDrive\Desktop\Bratsimages_normalized_onlybrain\images\image_'+str(img)+'.npy', temp_combined_images)
#         np.save(r'C:\Users\kesch\OneDrive\Desktop\Bratsimages_normalized_onlybrain\masks\mask_'+str(img)+'.npy', temp_mask)
        
#     else:
#         print("I am useless")

# # import splitfolders  # or import split_folders

# # input_folder = r'C:\Users\kesch\OneDrive\Desktop\2Brats2020'
# # output_folder = r'C:\Users\kesch\OneDrive\Desktop\BratsSeg2'
# # # Split with a ratio.
# # # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
# # splitfolders.ratio(input_folder, output=output_folder, seed=0, ratio=(.80, .2), group_prefix=None) # default values

# # def load_img(img_dir, img_list):
# #     images=[]
# #     for i, image_name in enumerate(img_list):    
# #         if (image_name.split('.')[1] == 'npy'):
            
# #             image = np.load(img_dir+ '/' + image_name)
                      
# #             images.append(image)
# #     images = np.array(images)
    
# #     return(images)




# # def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):

# #     L = len(img_list)

# #     # keras needs the generator infinite, so we will use while true  
# #     while True:

# #         batch_start = 0
# #         batch_end = batch_size

# #         while batch_start < L:
# #             limit = min(batch_end, L)
                       
# #             X = load_img(img_dir, img_list[batch_start:limit])
# #             Y = load_img(mask_dir, mask_list[batch_start:limit])

# #             yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

# #             batch_start += batch_size   
# #             batch_end += batch_size

# # ###########################################

# # # Test the generator

# # from matplotlib import pyplot as plt
# # import random

# # train_img_dir = R"C:\Users\kesch\OneDrive\Desktop\BratsSeg2\train\images"
# # train_mask_dir = R"C:\Users\kesch\OneDrive\Desktop\BratsSeg2\train\masks"
# # train_img_list=os.listdir(train_img_dir)
# # train_mask_list = os.listdir(train_mask_dir)

# # batch_size = 2

# # train_img_datagen = imageLoader(train_img_dir, train_img_list, 
# #                                 train_mask_dir, train_mask_list, batch_size)

# # # Verify generator.... In python 3 next() is renamed as __next__()
# # img, msk = train_img_datagen.__next__()


# # img_num = random.randint(0,img.shape[0]-1)
# # test_img=img[img_num]
# # test_mask=msk[img_num]
# # test_mask=np.argmax(test_mask, axis=3)

# # n_slice=random.randint(0, test_mask.shape[2])
# # plt.figure(figsize=(12, 8))

# # plt.subplot(221)
# # plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
# # plt.title('Image flair')
# # plt.subplot(222)
# # plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
# # plt.title('Image t1ce')
# # plt.subplot(223)
# # plt.imshow(test_img[:,:,n_slice, 3], cmap='gray')
# # plt.title('Image t2')
# # plt.subplot(224)
# # plt.imshow(test_mask[:,:,n_slice])
# # plt.title('Mask')
# # plt.show()


# #Change val to train
# images = glob.glob(R"C:\Users\kesch\OneDrive\Desktop\BratsSeg5\val\masks\*.npy")

# for i in images:
#     img = np.load(i)
#     img = img[:,:,:,1] + img[:,:,:,2] + img[:,:,:,3]
#     i = i.replace("BratsSeg5", "BratsSegBinary")
#     np.save(i, img)

import glob
import numpy as np

images = glob.glob(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\*.npy")
masks = glob.glob(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\masks\*.npy")

for i in images:
    msg = i
    msg = msg.replace("BrainTumorSeg\\val","Bratsimages_normalized_onlybrain")
    img = np.load(msg)
    np.save(i.replace("BrainTumorSeg", "Full_Segmentation"), img)
    
for j in masks:
    msk = np.load(j.replace("BrainTumorSeg\\val","Bratsimages_normalized_onlybrain"))
    np.save(j.replace("BrainTumorSeg", "Full_Segmentation"), msk)

