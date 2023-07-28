import glob
import os.path
from numpy import ndarray
import numpy as np
import nibabel as nib
from tqdm import tqdm
from glob import glob
import gzip
import shutil


def calc_z_score(img: ndarray) -> ndarray:
    """
    Standardize the image data using the zscore (z = (x-μ)/σ).

    :param img: Image data with shape components of (width, height, depth).
    :return:
    """
    avg_pixel_value = np.sum(img) / np.count_nonzero(img)
    sd_pixel_value = np.std(img[np.nonzero(img)])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i, j, k] != 0:
                    img[i, j, k] = (img[i, j, k] - avg_pixel_value) / sd_pixel_value

    return img


def change_mask_shape(mask: ndarray):
    if mask.shape == (128, 128, 128, 4):
        raise ValueError(
            f"Mask shape is already (128, 128, 128, 4)")

    new_mask = np.zeros((128, 128, 128, 4))
    for i in range(128):
        for j in range(128):
            for k in range(128):
                new_mask[i, j, k, mask[i, j, k]] = 1

    return new_mask


def normalize_mri_data(t1: ndarray, t1ce: ndarray, t2: ndarray, flair: ndarray, mask: ndarray) \
        -> tuple[ndarray, ndarray]:
    """
    Normalize the MRI data from the dataset using the z-score of the MRI data.

    :param t1: T1-Weighted MRI data
    :param t1ce: T1-Weighted Contrast Enhanced MRI data
    :param t2: T2-Weighted MRI data
    :param flair: Flair MRI data
    :param mask: Segmented mask data
    :return: Stacked MRI data and segmented mask data
    """

    t2 = t2[56:184, 56:184, 13:141]
    t2 = t2.reshape(-1, t2.shape[-1]).reshape(t2.shape)
    t2 = calc_z_score(t2)
    t1ce = t1ce[56:184, 56:184, 13:141]
    t1ce = t1ce.reshape(-1, t1ce.shape[-1]).reshape(t1ce.shape)
    t1ce = calc_z_score(t1ce)

    flair = flair[56:184, 56:184, 13:141]
    flair = flair.reshape(-1, flair.shape[-1]).reshape(flair.shape)
    flair = calc_z_score(flair)

    t1 = t1[56:184, 56:184, 13:141]
    t1 = t1.reshape(-1, t1.shape[-1]).reshape(t1.shape)
    t1 = calc_z_score(t1)

    mask = mask.astype(np.uint8)
    mask[mask == 4] = 3
    mask = mask[56:184, 56:184, 13:141]

    data = np.stack([flair, t1ce, t1, t2], axis=3)

    mask = change_mask_shape(mask)

    return data, mask


def get_mri_data_from_directory(patient_directory: str, t1: str, t1ce: str, t2: str, flair: str, mask: str) \
        -> type[ndarray, ndarray]:
    t1_data = nib.load(os.path.join(patient_directory, t1)).get_fdata()
    t1ce_data = nib.load(os.path.join(patient_directory, t1ce)).get_fdata()
    t2_data = nib.load(os.path.join(patient_directory, t2)).get_fdata()
    flair_data = nib.load(os.path.join(patient_directory, flair)).get_fdata()
    mask_data = nib.load(os.path.join(patient_directory, mask)).get_fdata()

    return normalize_mri_data(t1_data, t1ce_data, t2_data, flair_data, mask_data)


def roi_crop(img: ndarray, mask: ndarray, model) -> tuple[ndarray, ndarray]:
    """
    Crop the image and mask using the binary mask model.

    :param img: image to crop
    :param mask: mask to crop
    :param model: model to create the binary mask from the image.
    :return:
    """
    img_input = np.expand_dims(img, axis=0)

    binary_mask = model.predict(img_input)
    binary_mask = binary_mask[0, :, :, :, 0]
    binary_mask = np.expand_dims(binary_mask, -1)
    loc = np.where(binary_mask == 1)
    thesh = 12
    a = max(0, np.amin(loc[0]) - thesh)
    b = min(128, np.amax(loc[0]) + thesh)
    c = max(0, np.amin(loc[1]) - thesh)
    d = min(128, np.amax(loc[1]) + thesh)
    e = max(0, np.amin(loc[2]) - thesh)
    f = min(128, np.amax(loc[2]) + thesh)

    img1 = np.concatenate((img[a:b, c:d, e:f], binary_mask[a:b, c:d, e:f]), axis=-1)
    return img1, mask[a:b, c:d, e:f]


def global_extraction(img: ndarray | list[ndarray], mask: ndarray | list[ndarray]) -> tuple[ndarray, ndarray]:
    """
    Crops the image to 48 x 48 x 128 x C.

    :param img:
    :param mask:
    """
    images = []
    masks = []
    for i in range(len(img)):
        img_temp = img[i]
        mask_temp = mask[i]

        or_r = np.random.randint(0, img_temp.shape[0] - 47)
        or_c = np.random.randint(0, img_temp.shape[1] - 47)
        or_d = np.random.randint(0, img_temp.shape[2] - 47)
        img_temp = img_temp[or_r:or_r + 48, or_c:or_c + 48, or_d:or_d + 48, :]
        mask_temp = mask_temp[or_r:or_r + 48, or_c:or_c + 48, or_d:or_d + 48, :]
        images.append(img_temp)
        masks.append(mask_temp)

    stacked_images = np.stack(images, axis=0)
    stacked_masks = np.stack(masks, axis=0)
    return stacked_images, stacked_masks


def mask_to_binary_mask(mask: ndarray) -> ndarray:
    """
    Convert the mask to a binary mask by checking the contents of the voxel.

    :param mask: mask to convert
    :return: a binary version of the mask
    """
    new_mask = np.zeros(mask.shape[:-1])
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i][j][k][0] != 1:
                    new_mask[i][j][k] = 1

    return new_mask


def decompress_patient_folders(patient_folders_directory: str, delete_archives=True) -> None:
    if not os.path.isdir(patient_folders_directory):
        raise FileNotFoundError("The patient folders directory is not a valid directory")

    for patient_folder_path in tqdm(os.listdir(patient_folders_directory)):
        patient_folder_path = os.path.join(patient_folders_directory, patient_folder_path)
        if not os.path.isdir(patient_folder_path):
            continue

        os.chdir(patient_folder_path)
        for mri_file in os.listdir(patient_folder_path):
            if not mri_file.endswith(".gz"):
                continue

            archive_path = os.path.abspath(mri_file)
            mri_file_name = os.path.basename(mri_file).rsplit('.', 1)[0]

            with gzip.open(archive_path, "rb") as archive_file, open(mri_file_name, "wb") as output_mri_file:
                shutil.copyfileobj(archive_file, output_mri_file)

            if delete_archives:
                # Remove the old compressed file
                os.remove(archive_path)


def random_train_test_split(data: list, train_split=0.8) -> tuple[list, list]:
    if train_split >= 1:
        raise ValueError("Train Split has to be less than 1")

    np.random.shuffle(data)
    data_len = len(data)

    return data[:int(data_len * train_split)], data[int(data_len * train_split):]

def create_dataset_from_patients_directory(patients_directory: str, output_dataset_directory: str) -> None:
    if not os.path.isdir(patients_directory):
        raise NotADirectoryError("The patients directory is not a valid directory")

    # All MRI data from .nii files as (patient_index, image_data, mask_data)
    all_mri_data = []
    print("Loading MRI Data")
    for patient_directory_name in (tqdm(os.listdir(patients_directory))):
        patient_path = os.path.join(patients_directory, patient_directory_name)

        if os.path.isdir(patient_path):
            mri_data = {}
            for file in os.listdir(patient_path):
                if "_t1." in file:
                    mri_data["t1"] = file
                elif "_t1ce." in file:
                    mri_data["t1ce"] = file
                elif "_t2." in file:
                    mri_data["t2"] = file
                elif "_flair." in file:
                    mri_data["flair"] = file
                elif "_seg." in file:
                    mri_data["mask"] = file

            # Extract the patient's id from the directory name
            patient_index = patient_directory_name[-5:]
            image, mask = get_mri_data_from_directory(patient_path, **mri_data)
            all_mri_data.append((patient_index, image, mask))

    print("Shuffling MRI Data into Train (80%) and Validation (20%) splits")
    train_mri_data, val_mri_data = random_train_test_split(all_mri_data)

    for category, data in [("train", train_mri_data), ("val", val_mri_data)]:
        output_images_directory = os.path.join(output_dataset_directory, category, "images")
        output_masks_directory = os.path.join(output_dataset_directory, category, "masks")

        if not os.path.isdir(output_images_directory):
            os.makedirs(output_images_directory)

        if not os.path.isdir(output_masks_directory):
            os.makedirs(output_masks_directory)

        print(f"Saving {category.title()} MRI Data")
        for patient_index, image, mask in tqdm(data):
            np.save(os.path.join(output_images_directory, f"image-{patient_index}.npy"), image)
            np.save(os.path.join(output_masks_directory, f"mask-{patient_index}.npy"), mask)


def create_binary_dataset_from_dataset(input_dataset: str, output_dataset_directory: str) -> None:
    if not os.path.isdir(input_dataset):
        raise NotADirectoryError("The dataset directory is not a valid directory")

    for category in ["train", "val"]:
        input_images_directory = os.path.join(input_dataset, category, "images")
        input_masks_directory = os.path.join(input_dataset, category, "masks")

        output_images_directory = os.path.join(output_dataset_directory, category, "images")
        output_masks_directory = os.path.join(output_dataset_directory, category, "masks")

        if not os.path.isdir(output_masks_directory):
            os.makedirs(output_masks_directory)

        print(f"Copying images over for the {category} category")
        shutil.copytree(input_images_directory, output_images_directory)

        print(f"Converting masks to binary masks for the {category} category")
        for mask_path in tqdm(glob(os.path.join(input_masks_directory, "*.npy"))):
            mask = np.load(mask_path)

            binary_mask = mask_to_binary_mask(mask)

            np.save(os.path.join(output_masks_directory, os.path.basename(mask_path)), binary_mask)


def create_cropped_dataset_from_dataset(dataset_directory: str, model, output_dataset_directory: str) -> None:
    if not os.path.isdir(dataset_directory):
        raise NotADirectoryError("The dataset directory is not a valid directory")

    for category in ["train", "val"]:
        all_images = glob(os.path.join(dataset_directory, category, 'images', "*.npy"))
        all_masks = glob(os.path.join(dataset_directory, category, 'masks', "*.npy"))

        if len(all_images) != len(all_masks):
            raise ValueError(f"There are not the same number of images and masks in the {category} category")

        output_images_directory = os.path.join(output_dataset_directory, category, "images")
        output_masks_directory = os.path.join(output_dataset_directory, category, "masks")

        if not os.path.isdir(output_images_directory):
            os.makedirs(output_images_directory)

        if not os.path.isdir(output_masks_directory):
            os.makedirs(output_masks_directory)

        print(f"Cropping and saving dataset for the {category} category")
        skipped = []
        for img_path, mask_path in tqdm(zip(all_images, all_masks), total=len(all_images)):
            img_data = np.load(img_path)
            mask_data = np.load(mask_path)

            cropped_image, cropped_mask = roi_crop(img_data, mask_data, model)

            if any([dimension < 48 for dimension in cropped_image.shape[:-1]]) or any([dimension < 48 for dimension in cropped_image.shape[:-1]]):
                skipped.append([os.path.basename(img_path), cropped_image.shape, os.path.basename(mask_path), cropped_mask.shape])
                continue

            np.save(os.path.join(output_images_directory, os.path.basename(img_path)), cropped_image)
            np.save(os.path.join(output_masks_directory, os.path.basename(mask_path)), cropped_mask)

        for img_name, img_shape, mask_name, mask_shape in skipped:
            print(f"Skipped Cropped Data: {img_name}:{img_shape}, {mask_name}:{mask_shape}")
