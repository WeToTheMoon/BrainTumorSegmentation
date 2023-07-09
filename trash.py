 # Print the dice score of each test sample
#
# test_imgs = glob(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\*.npy")
#
# dice_coef_final = 0
#
# for i in test_imgs:
#     test_img = np.load(i)[..., :-1]
#     i = i.replace("images", "masks")
#     i = i.replace("image", "mask")
#     test_mask = np.load(i)
#     test_mask = np.expand_dims(test_mask, axis=0)
#     test_img_input = np.expand_dims(test_img, axis=0)
#     test_img_input, test_mask = global_extraction(test_img_input, test_mask)
#     region_based1 = model_whole.predict(test_img_input)
#     test_img_input = np.concatenate((test_img_input, region_based1), axis=-1)
#     test_prediction = model.predict(test_img_input)
#     dice_coef1 = dice_coef_multilabel(test_mask, test_prediction, 3)
#     print(dice_coef1)
#
# ##################################################################################################################
# # Print out the entire mask prediction
#
# img = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\image_120.npy")[..., :-1]
# msk = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\masks\mask_120.npy")
#
# img = np.expand_dims(img, axis=0)
# msk = np.expand_dims(msk, axis=0)
# img1 = img[:, :48, :48, :, :]
# img2 = img[:, 38:87, :48, :, :]
# img3 = img[:, :48, 43:92, :, :]
# img4 = img[:, 38:87, 43:92, :, :]
#
# region_based1 = model_whole.predict(img1)
# region_based2 = model_whole.predict(img2)
# region_based3 = model_whole.predict(img3)
# region_based4 = model_whole.predict(img4)
#
# img1 = np.concatenate((img1, region_based1), axis=-1)
# img2 = np.concatenate((img2, region_based2), axis=-1)
# img3 = np.concatenate((img3, region_based3), axis=-1)
# img4 = np.concatenate((img4, region_based4), axis=-1)
#
# test_prediction1 = model.predict([img1])
# test_prediction2 = model.predict([img2])
# test_prediction3 = model.predict([img3])
# test_prediction4 = model.predict([img4])
#
# img_final = np.zeros((1, 86, 91, 128, 4))
#
# img_final[:, :48, :48, :, :] = test_prediction1
# img_final[:, 38:87, :48, :, :] = test_prediction2
# img_final[:, :48, 43:92, :, :] = test_prediction3
# img_final[:, 38:87, 43:92, :, :] = test_prediction4
#
# img_final = np.argmax(img_final, axis=4)
# msk = np.argmax(msk, axis=4)
#
# print(img_final[0, 50, :, :])
#
# for i in range(50, 70):
#     n_slice = i
#     plt.figure(figsize=(12, 8))
#     plt.subplot(231)
#     plt.title('Flair')
#     plt.imshow(img[0, :, :, n_slice, 0], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(232)
#     plt.title('T1CE')
#     plt.imshow(img[0, :, :, n_slice, 1], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(233)
#     plt.title('T1')
#     plt.imshow(img[0, :, :, n_slice, 2], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(234)
#     plt.title('T2')
#     plt.imshow(img[0, :, :, n_slice, 3], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(235)
#     plt.title('Ground Truth')
#     plt.imshow(msk[0, :, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(236)
#     plt.title('Predicted Mask')
#     plt.imshow(img_final[0, :, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.show()

###############################################################################################################

# img_orig = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\image_239.npy")[..., :-1]
# msk = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\masks\mask_239.npy")

# img_orig  = np.expand_dims(img_orig, axis=0)
# msk  = np.argmax(np.expand_dims(msk, axis=0), axis = -1)
# img,msk = global_extraction(img,msk)
# region_based = model_whole.predict(img)
# img8 = np.concatenate((img, region_based), axis = -1)
# test_prediction8 = model.predict([img8])

# img1 = np.flip(img,axis=1)
# region_based1 = model_whole.predict(img1)
# img1 = np.concatenate((img1, region_based1), axis = -1)
# test_prediction1 = model.predict([img1])
# test_prediction1 = np.flip(test_prediction1,axis=(1))

# img2 = np.flip(img,axis=2)
# region_based2 = model_whole.predict(img2)
# img2 = np.concatenate((img2, region_based2), axis = -1)
# test_prediction2 = model.predict([img2])
# test_prediction2 = np.flip(test_prediction2,axis=(2))

# img3 = np.flip(img,axis=3)
# region_based3 = model_whole.predict(img3)
# img3 = np.concatenate((img3, region_based3), axis = -1)
# test_prediction3 = model.predict([img3])
# test_prediction3 = np.flip(test_prediction3,axis=(3))

# img4 = np.flip(img,axis=(1,2))
# region_based4 = model_whole.predict(img4)
# img4 = np.concatenate((img4, region_based4), axis = -1)
# test_prediction4 = model.predict([img4])
# test_prediction4 = np.flip(test_prediction4,axis=(1,2))

# img5 = np.flip(img,axis=(2,3))
# region_based5 = model_whole.predict(img5)
# img5 = np.concatenate((img5, region_based5), axis = -1)
# test_prediction5 = model.predict([img5])
# test_prediction5 = np.flip(test_prediction5,axis=(2,3))

# img6 = np.flip(img,axis=(3,1))
# region_based6 = model_whole.predict(img6)
# img6 = np.concatenate((img6, region_based6), axis = -1)
# test_prediction6 = model.predict([img6])
# test_prediction6 = np.flip(test_prediction6,axis=(1,3))

# img7 = np.flip(img,axis=(1,2,3))
# region_based7 = model_whole.predict(img7)
# img7 = np.concatenate((img7, region_based7), axis = -1)
# test_prediction7 = model.predict([img7])
# test_prediction7 = np.flip(test_prediction7,axis=(1,2,3))

# test_prediction = (test_prediction8 + test_prediction1 + test_prediction2 + test_prediction3 + test_prediction4 + test_prediction5 + test_prediction6 + test_prediction7)/8


# confidence = np.zeros((1,48,48,128))

# for k in range(10,115):
#     if np.sum(test_prediction[:,:,:,k] > 0):
#         for j in range(48):
#             for l in range(48):
#                 confidence[:,l,j,k] = tf.math.reduce_logsumexp((test_prediction[0,l,j,k]))

# for i in range(10,115):
#         print(np.sum(confidence[:,:,:,i]))
# # + np.amax(test_prediction[0,l,j,k])

# test_prediction = np.argmax(test_prediction, axis=4)
# test_prediction8 = np.argmax(test_prediction8, axis=4)
# msk = np.argmax(msk, axis=4)


# img_orig = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\image_239.npy")[..., :-1]
# msk = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\masks\mask_239.npy")

# img_pred, msk, confidence = put_together(img_orig,msk)

# img_orig1 = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\image_33.npy")[..., :-1]
# msk1 = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\masks\mask_33.npy")

# img_pred1, msk1, confidence1 = put_together(img_orig1,msk1)

# img_orig2 = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\image_21.npy")[..., :-1]
# msk2 = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\masks\mask_21.npy")

# img_pred2, msk2, confidence2 = put_together(img_orig2,msk2)

# img_orig3 = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\image_41.npy")[..., :-1]
# msk3 = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\masks\mask_41.npy")

# img_pred3, msk3, confidence3 = put_together(img_orig3,msk3)

# img_orig4 = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\image_120.npy")[..., :-1]
# msk4 = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\masks\mask_120.npy")

# img_pred4, msk4, confidence4 = put_together(img_orig4,msk4)


# img_orig = np.expand_dims(img_orig, axis=0)
# img_orig1 = np.expand_dims(img_orig, axis=0)
# img_orig2 = np.expand_dims(img_orig, axis=0)
# img_orig3 = np.expand_dims(img_orig, axis=0)
# img_orig4 = np.expand_dims(img_orig, axis=0)

# for i in range(128):
#     n_slice = i
#     plt.figure(figsize=(12, 8))
#     plt.subplot(231)
#     plt.title('Flair')
#     plt.imshow(img_orig[0, :, :, n_slice,0], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(232)
#     plt.title('Confidence')
#     plt.imshow(confidence[0, :, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(233)
#     plt.title('Ground Truth')
#     plt.imshow(msk[0, :, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(234)
#     plt.title('Predicted Mask')
#     plt.imshow(img_pred[0,:, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.show()

# for i in range(128):
#     n_slice = i
#     plt.figure(figsize=(12, 8))
#     plt.subplot(231)
#     plt.title('Flair')
#     plt.imshow(img_orig1[0, :, :, n_slice,0], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(232)
#     plt.title('Confidence')
#     plt.imshow(confidence1[0, :, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(233)
#     plt.title('Ground Truth')
#     plt.imshow(msk1[0, :, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(234)
#     plt.title('Predicted Mask')
#     plt.imshow(img_pred1[0,:, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.show()

# for i in range(128):
#     n_slice = i
#     plt.figure(figsize=(12, 8))
#     plt.subplot(231)
#     plt.title('Flair')
#     plt.imshow(img_orig2[0, :, :, n_slice,0], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(232)
#     plt.title('Confidence')
#     plt.imshow(confidence2[0, :, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(233)
#     plt.title('Ground Truth')
#     plt.imshow(msk2[0, :, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(234)
#     plt.title('Predicted Mask')
#     plt.imshow(img_pred2[0,:, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.show()

# for i in range(128):
#     n_slice = i
#     plt.figure(figsize=(12, 8))
#     plt.subplot(231)
#     plt.title('Flair')
#     plt.imshow(img_orig3[0, :, :, n_slice,0], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(232)
#     plt.title('Confidence')
#     plt.imshow(confidence3[0, :, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(233)
#     plt.title('Ground Truth')
#     plt.imshow(msk3[0, :, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(234)
#     plt.title('Predicted Mask')
#     plt.imshow(img_pred3[0,:, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.show()

# for i in range(128):
#     n_slice = i
#     plt.figure(figsize=(12, 8))
#     plt.subplot(231)
#     plt.title('Flair')
#     plt.imshow(img_orig4[0, :, :, n_slice,0], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(232)
#     plt.title('Confidence')
#     plt.imshow(confidence4[0, :, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(233)
#     plt.title('Ground Truth')
#     plt.imshow(msk4[0, :, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(234)
#     plt.title('Predicted Mask')
#     plt.imshow(img_pred4[0,:, :, n_slice])
#     plt.grid(False)
#     plt.axis('off')
#     plt.show()


# img = np.load(r"C:\Users\kesch\OneDrive\Desktop\BrainTumorSeg\val\images\image_0.npy")[..., :-1]
#
# print(img.shape)
# for i in range(128):
#     plt.figure(figsize=(12, 8))
#     n_slice = i
#     plt.subplot(231)
#     plt.title('Flair')
#     plt.imshow(img[:, :, n_slice, 0], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(232)
#     plt.title('t1gd')
#     plt.imshow(img[:, :, n_slice, 1], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(233)
#     plt.title('t1')
#     plt.imshow(img[:, :, n_slice, 2], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')
#     plt.subplot(234)
#     plt.title('t2')
#     plt.imshow(img[:, :, n_slice, 3], cmap='gray')
#     plt.grid(False)
#     plt.axis('off')

# img_num = 144
#
# test_img = np.load(
#     r"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\val\images\image_" + str(img_num) + ".npy")
#
# test_mask = np.load(
#     r"C:\Users\kesch\OneDrive\Desktop\BratsSegBinary\val\masks\mask_" + str(img_num) + ".npy")
#
# # test_mask_argmax = np.argmax(test_mask, axis=3)
#
# test_img_input = np.expand_dims(test_img, axis=0)
# test_prediction = model.predict(test_img_input)
# test_prediction = test_prediction[0, :, :, :, 0]
#
# for i in range(128):
#     n_slice = i
#     plt.figure(figsize=(12, 8))
#     plt.subplot(231)
#     plt.title('Testing Image')
#     plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
#     plt.subplot(232)
#     plt.title('Testing Label')
#     plt.imshow(test_mask[:, :, n_slice])
#     plt.subplot(233)
#     plt.title('Prediction on test image')
#     plt.imshow(test_prediction[:, :, n_slice])
#     plt.show()
#
# def uncert(img):
#     # given img shape is 1, 48, 48, 128, 4
#     region_based = model_whole.predict(img)
#     img8 = np.concatenate((img, region_based), axis=-1)
#     test_prediction8 = model.predict([img8])
#
#     img1 = np.flip(img, axis=1)
#     region_based1 = model_whole.predict(img1)
#     img1 = np.concatenate((img1, region_based1), axis=-1)
#     test_prediction1 = model.predict([img1])
#     test_prediction1 = np.flip(test_prediction1, axis=1)
#
#     img2 = np.flip(img, axis=2)
#     region_based2 = model_whole.predict(img2)
#     img2 = np.concatenate((img2, region_based2), axis=-1)
#     test_prediction2 = model.predict([img2])
#     test_prediction2 = np.flip(test_prediction2, axis=2)
#
#     img3 = np.flip(img, axis=3)
#     region_based3 = model_whole.predict(img3)
#     img3 = np.concatenate((img3, region_based3), axis=-1)
#     test_prediction3 = model.predict([img3])
#     test_prediction3 = np.flip(test_prediction3, axis=3)
#
#     img4 = np.flip(img, axis=(1, 2))
#     region_based4 = model_whole.predict(img4)
#     img4 = np.concatenate((img4, region_based4), axis=-1)
#     test_prediction4 = model.predict([img4])
#     test_prediction4 = np.flip(test_prediction4, axis=(1, 2))
#
#     img5 = np.flip(img, axis=(2, 3))
#     region_based5 = model_whole.predict(img5)
#     img5 = np.concatenate((img5, region_based5), axis=-1)
#     test_prediction5 = model.predict([img5])
#     test_prediction5 = np.flip(test_prediction5, axis=(2, 3))
#
#     img6 = np.flip(img, axis=(3, 1))
#     region_based6 = model_whole.predict(img6)
#     img6 = np.concatenate((img6, region_based6), axis=-1)
#     test_prediction6 = model.predict([img6])
#     test_prediction6 = np.flip(test_prediction6, axis=(1, 3))
#
#     img7 = np.flip(img, axis=(1, 2, 3))
#     region_based7 = model_whole.predict(img7)
#     img7 = np.concatenate((img7, region_based7), axis=-1)
#     test_prediction7 = model.predict([img7])
#     test_prediction7 = np.flip(test_prediction7, axis=(1, 2, 3))
#
#     test_prediction = (test_prediction8 + test_prediction1 + test_prediction2 + test_prediction3 + test_prediction4
#                        + test_prediction5 + test_prediction6 + test_prediction7) / 8
#     confidence = np.zeros((1, 48, 48, 128))
#
#     for k in range(128):
#         for j in range(48):
#             for q in range(48):
#                 confidence[:, q, j, k] = tf.math.reduce_logsumexp(-1 * (test_prediction[0, q, j, k]))
#     return confidence
#
#
# def put_together(img, msk):
#     img = np.expand_dims(img, axis=0)
#     msk = np.expand_dims(msk, axis=0)
#
#     a = 0
#     b = 48
#     c = 0
#     d = 48
#
#     img_final = np.zeros(img.shape)
#     uncertainty = np.zeros((img.shape[0], img.shape[1], img.shape[2], img.shape[3]))
#
#     for i in range((img.shape[2] // 48) + 1):
#         a = 0
#         b = 48
#         for j in range((img.shape[1] // 48) + 1):
#             img_temp = img[:, a:b, c:d, :, :]
#             uncertainty[:, a:b, c:d, :] = uncert(img_temp)
#             region_based = model_whole.predict(img_temp)
#
#             img_temp = np.concatenate((img_temp, region_based), axis=-1)
#             test_prediction = model.predict([img_temp])
#             img_final[:, a:b, c:d, :, :] = test_prediction
#             if b + 48 < img.shape[1]:
#                 a += 48
#                 b += 48
#             else:
#                 b = img.shape[1]
#                 a = b - 48
#
#         if d + 48 < img.shape[2]:
#             c += 48
#             d += 48
#         else:
#             d = img.shape[2]
#             c = d - 48
#
#     img_final = np.argmax(img_final, axis=4)
#     msk = np.argmax(msk, axis=4)
#     return img_final, msk, uncertainty