# BrainTumorSegmentation

## Introduction

## Abstract

## Setup
### Getting Started
1. Clone the repository
    ```
    git clone https://github.com/WeToTheMoon/BrainTumorSegmentation.git
    ```
2. Download the datasets being used
    - [BraTS2019]()
    - [BraTS2020]()
    - [BraTS2021]()
### Dependencies
It is recommended to create a conda or venv when working with this project. A full list of requirements can be found in the [requirements.txt](requirements.txt) file. Python 3.10.12 was used.

### Creating Dataset
Datasets consist of two main types, binary and cropped datasets. The binary dataset is created using the binary model and is used in order to create the cropped dataset which is used by the multiclass model. All of these datasets are wrapped using the `MRIDataset` class in order to easily validate and access its data.


In order to create the dataset, [a helper method](utils/dataset_helpers.py) has been provided. Simply call the method with the location of the previously downloaded dataset and the desired output path.
```
from utils.dataset_helpers import create_new_dataset

input_dataset_path = ...
output_dataset_path = ...

create_new_dataset(input_dataset_path, output_dataset_path)
```

### Results
After creating the dataset, the multiclass model can be run
```
python train_binary_model --dataset_dir <cropped_dataset_path> --weights <location_to_save_model_weights>
```
or
```
from train_binary_model import train as train_multiclass_model

cropped_dataset_path = ...
multiclass_model_weights = ...
train_multiclass_model(cropped_dataset_path, multiclass_model_weights)
```
