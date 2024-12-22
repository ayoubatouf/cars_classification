#  Cars class prediction

## Overview

In this project, we developed an extended version of the ResNet-50 architecture and trained it from scratch on the Stanford Cars dataset (16,185 images) `(https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder)` to predict the classes of 196 different car models.


## Usage

Download the dataset and place it in the `/raw` path (ensure that 70% of the data is used for training and 30% for testing). To train the model from scratch, execute `/training/main.py`. To perform inference on a new image, specify the image path in `/config/paths.py` and execute `/inference/inference.py`. This will use the pretrained model and `/config/class_labels.json` to predict the class of the input car.

## Example

![](car.jpg)

input : 
- raw/car_data/test/Mercedes-Benz SL-Class Coupe 2009/02615.jpg

output : 

- Predicted class index: 71
- Predicted class: Mercedes-Benz SL-Class Coupe 2009


