import glob
import os.path

from numpy import ndarray
import numpy as np
import nibabel as nib
from tqdm import tqdm, trange

import glob


def calc_z_score(img: ndarray, img_height: int, img_width: int, img_depth: int) -> ndarray:
    avg_pixel_value = np.sum(img) / np.count_nonzero(img)
    sd_pixel_value = np.std(img[np.nonzero(img)])

    for i in range(img_width):
        for j in range(img_height):
            for k in range(img_depth):
                if img[i, j, k] != 0:
                    img[i, j, k] = (img[i, j, k] - avg_pixel_value) / sd_pixel_value

    return img


def normalize_mri_data(t1: ndarray, t1ce: ndarray, t2: ndarray, flair: ndarray, mask: ndarray) \
        -> tuple[ndarray, ndarray]:
    t2 = t2[56:184, 56:184, 13:141].reshape(-1, t2.shape[-1]).reshape(t2.shape)
    t2 = calc_z_score(t2)

    t1ce = t1ce[56:184, 56:184, 13:141].reshape(-1, t1ce.shape[-1]).reshape(t1ce.shape)
    t1ce = calc_z_score(t1ce)

    flair = flair[56:184, 56:184, 13:141].reshape(-1, flair.shape[-1]).reshape(flair.shape)
    flair = calc_z_score(flair)

    t1 = t1[56:184, 56:184, 13:141].reshape(-1, t1.shape[-1]).reshape(t1.shape)
    t1 = calc_z_score(t1)

    mask = mask.astype(np.uint8)
    mask[mask == 4] = 3
    mask = mask[56:184, 56:184, 13:141]

    data = np.stack([flair, t1ce, t1, t2], axis=3)

    return data, mask


def load_dataset_from_patient_data(patients_directory: str, output_dataset: str,
                                   train_dir: str = "train", val_dir: str = "val") -> None:
    if not os.path.isdir(patients_directory):
        raise FileNotFoundError("The Patients directory isn't a valid directory")

    all_patients = [f"{patients_directory}/{patient}" for patient in os.listdir(patients_directory) if
                    os.path.isdir(f"{patients_directory}/{patient}")]

    # Randomly shuffle the images
    np.random.shuffle(all_patients)

    train_patients = all_patients[:int(len(all_patients) * .80)]
    val_patients = all_patients[int(len(all_patients) * .80):]

    for patient_list, save_directory in [(train_patients, train_dir), (val_patients, val_dir)]:
        print(f"Collecting dataset for the {save_directory} directory")
        i = 0
        for patient_path in tqdm(patient_list):
            i += 1
            mri_data = {}
            for sub_file in os.listdir(patient_path):
                file_path = f"{patient_path}/{sub_file}"
                if "_t1." in file_path:
                    mri_data["t1"] = nib.load(file_path).get_fdata()
                elif "_t1ce." in file_path:
                    mri_data["t1ce"] = nib.load(file_path).get_fdata()
                elif "_t2." in file_path:
                    mri_data["t2"] = nib.load(file_path).get_fdata()
                elif "_flair." in file_path:
                    mri_data["flair"] = nib.load(file_path).get_fdata()
                elif "_seg." in file_path:
                    mri_data["mask"] = nib.load(file_path).get_fdata()

            if len(mri_data) != 5:
                raise FileExistsError("Missing one or more required MRI modalities")

            data, mask = normalize_mri_data(**mri_data)

            np.save(f"{output_dataset}/{save_directory}/images/image-{i}.npy", data)
            np.save(f"{output_dataset}/{save_directory}/masks/mask-{i}.npy", mask)


def roi_crop(img: ndarray, mask: ndarray, model) -> tuple[ndarray, ndarray]:
    img_input = np.expand_dims(img, axis=0)

    binary_mask = model.predict(img_input)
    binary_mask = binary_mask[0, :, :, :, 0]

    loc = np.where(binary_mask == 1)
    a = max(0, np.amin(loc[0]) - 12)
    b = min(128, np.amax(loc[0]) + 12)
    c = max(0, np.amin(loc[1]) - 12)
    d = min(128, np.amax(loc[1]) + 12)
    e = max(0, np.amin(loc[2]) - 12)
    f = min(128, np.amax(loc[2]) + 12)
    return np.concatenate((img[a:b, c:d, e:f], np.expand_dims(binary_mask[a:b, c:d, e:f], -1)), axis=-1), mask[a:b, c:d, e:f]


def create_cropped_dataset_from_roi(model, input_dataset, full_mask_dir, output_dataset):
    train_imgs = glob.glob(input_dataset + "\\train\images\*.npy")
    train_msks = glob.glob(input_dataset + "\\train\masks\*.npy")
    val_imgs = glob.glob(input_dataset + "\\val\images\*.npy")
    val_msks = glob.glob(input_dataset + "\\val\masks\*.npy")

    for i in zip(train_imgs, train_msks):
        for j in glob.glob(full_mask_dir + "\\train\masks\*.npy"):
            if i[1].split('\\')[-1] == j.split('\\')[-1]:
                mask_loc = j
        img = np.load(i[0])
        msk = np.load(mask_loc)
        cropped_img, cropped_msk = roi_crop(img, msk, model)
        print(cropped_img.shape)
        np.save(output_dataset + "\\train\images\\" + i[0].split('\\')[-1], cropped_img)
        np.save(output_dataset + "\\train\masks\\" + i[1].split('\\')[-1], cropped_msk)

    for i in zip(val_imgs, val_msks):
        for j in glob.glob(full_mask_dir + "\\val\masks\*.npy"):
            if i[1].split('\\')[-1] == j.split('\\')[-1]:
                mask_loc = j
        img = np.load(i[0])
        msk = np.load(j)
        cropped_img, cropped_msk = roi_crop(img, msk, model)
        np.save(output_dataset + "\\val\images\\" + i[0].split('\\')[-1], cropped_img)
        np.save(output_dataset + "\\val\masks\\" + i[1].split('\\')[-1], cropped_msk)


#
# def create_cropped_dataset_from_roi(input_dataset: str, input_train_dir: str, input_val_dir: str, output_dataset: str,
#                                     model, output_train_dir: str = "train", output_val_dir: str = "val") -> None:
#     if not all([v in os.listdir(input_dataset) for v in [input_train_dir, input_val_dir]]):
#         raise FileNotFoundError("Missing either the train or validation directory from within the dataset directory")
#
#     if not all([v in os.listdir(os.path.join(input_dataset, input_train_dir))
#                 and v in os.listdir(os.path.join(input_dataset, input_val_dir)) for v in ["images", "masks"]]):
#         raise FileNotFoundError(
#             "The training and/or validation directories are missing the 'images' and/or 'masks' directories")
#
#     for input_section_path, output_section_path in [(input_train_dir, output_train_dir),
#                                                     (input_val_dir, output_val_dir)]:
#         base_section_path = os.path.join(input_dataset, input_section_path)
#         input_images_dir = os.path.join(base_section_path, "images")
#         input_masks_dir = os.path.join(base_section_path, "masks")
#
#         if len(os.listdir(input_images_dir)) != len(os.listdir(input_masks_dir)):
#             raise ValueError("The images and masks directories don't have the same lengths")
#
#         print(f"Collecting and converting dataset for the {input_images_dir} directory")
#         # The images and masks directories should have the same lengths
#         for i in trange(len(os.listdir(input_images_dir))):
#             input_image = np.load(os.path.join(input_images_dir, f"image_{i}.npy"))
#             input_mask = np.load(os.path.join(input_masks_dir, f"mask_{i}.npy"))
#
#             cropped_image, cropped_mask = roi_crop(input_image, input_mask, model)
#
#             np.save(os.path.join(output_dataset, output_section_path, "images", f"image-cropped-{i}.npy"),
#                     cropped_image)
#             np.save(os.path.join(output_dataset, output_section_path, "masks", f"mask-cropped-{i}.npy"), cropped_mask)
#
