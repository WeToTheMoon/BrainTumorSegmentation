import glob
import os.path

from numpy import ndarray
import numpy as np
import nibabel as nib
from tqdm import tqdm, trange

from glob import glob


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
    binary_mask = np.expand_dims(binary_mask, -1)
    loc = np.where(binary_mask == 1)
    a = max(0, np.amin(loc[0]) - 12)
    b = min(128, np.amax(loc[0]) + 12)
    c = max(0, np.amin(loc[1]) - 12)
    d = min(128, np.amax(loc[1]) + 12)
    e = max(0, np.amin(loc[2]) - 12)
    f = min(128, np.amax(loc[2]) + 12)

    img1 = np.concatenate((img[a:b, c:d, e:f], binary_mask[a:b, c:d, e:f]), axis=-1)
    return img1, mask[a:b, c:d, e:f]


def create_cropped_dataset_from_roi(input_dataset: str, output_dataset: str, train_dir: str = "train", val_dir: str = "val", model=None):
    train_images = glob(f"{input_dataset}/{train_dir}/images/*.npy")
    train_masks = glob(f"{input_dataset}/{train_dir}/masks/*.npy")
    val_images = glob(f"{input_dataset}/{val_dir}/images/*.npy")
    val_masks = glob(f"{input_dataset}/{val_dir}/masks/*.npy")

    print("Processing Training Images and Masks")
    for img_path, mask_path in tqdm(zip(train_images, train_masks)):
        img = np.load(img_path)
        mask = np.load(mask_path)

        cropped_image, cropped_mask = roi_crop(img, mask, model)
        print(cropped_mask.shape)
        np.save(os.path.join(output_dataset, train_dir, "images", f"{os.path.basename(img_path)}"), cropped_image)
        np.save(os.path.join(output_dataset, train_dir, "masks", f"{os.path.basename(mask_path)}"), cropped_mask)

    print("Processing Validation Images and Masks")
    for img_path, mask_path in tqdm(zip(val_images, val_masks)):
        img = np.load(img_path)
        mask = np.load(mask_path)

        cropped_image, cropped_mask = roi_crop(img, mask, model)
        np.save(os.path.join(output_dataset, val_dir, "images", f"{os.path.basename(img_path)}"), cropped_image)
        np.save(os.path.join(output_dataset, val_dir, "masks", f"{os.path.basename(mask_path)}"), cropped_mask)

#
