[![pl](https://img.shields.io/badge/język-PL-red.svg)](https://github.com/pzemla/Image-classification-using-CNN/blob/main/README.pl.md)
# Image classification using CNN

**Dependencies**

Python 3.9.13

matplotlib 3.8.3

notebook 7.1.2

numpy 1.24.1

pandas  2.2.1

scikit-learn 1.4.1.post1 Python 3.9.13

torch 2.2.1+cu118

**How to run**
1. Download dataset from https://www.kaggle.com/datasets/flo2607/traffic-signs-classification
2. Put directories train and test_all together in directory with convolutional.ipynb
3. Run convolutional.ipynb in Jupyter Notebook

## Overview

The goal of this project is to build a convolutional neural network (CNN) to classify 64x64x3 images into one of 50 predefined classes. CNN is implemented using Python with the Pytorch library. The model is trained on a labeled dataset containing images from different categories to accurately predict the class labels of images. This project serves as an educational exercise and practical application of deep learning techniques to image classification tasks. 

The number of images per class in the training set is usually between 300 and 400 images, except for two classes, one of which has around 200 images and the other has less than 100, as shown on histogram below.

![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/af85d2ef-a690-48db-88c6-d627ccc7958d)


Below are sample photos of images from 3 different classes: 

Batteries

![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/4bd53ef6-e0bc-4b7f-9e30-0b5a2174ccfe)
![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/085ab246-5d8b-4726-8e7b-6e70bb868ebc)
![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/59320a5b-1e69-4a05-92f6-afe9b5442d4e)

Turtles 

![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/cdf39672-2bf8-475b-abb7-01140a03260f)
![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/5acd752a-0376-43f1-8ba0-cb1ccc47a90f)
![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/d6a8450b-59f2-41b1-b449-dc0627cc7280)

Towels 

![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/5d383acb-46d5-4d2c-bf3c-5e55cbabda57)
![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/74cdf248-3258-42b0-b141-c7f092cdd2e4)
![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/8caf964f-1856-40b8-bac7-67119f141bf6)

# CNN structure
A Convolutional Neural Network (CNN) is a type of artificial neural network designed to analyze visual data, like images. It operates by passing input images through layers of convolutional filters, which extract features like edges and textures. These features are then progressively combined and analyzed through additional layers, enabling the network to learn complex patterns and make predictions, such as object recognition or image classification.

|Layer|Description|Input|Output|
| ------------- | ------------- | ------------- | ------------- |
|Convolutional|Stride=1, kernel=3, padding=1|64x64x3|64x64x32|
|Batch normalization|Normalization of output from the convolutional layer|64x64x32|64x64x32|
|Relu|Activation function|64x64x32|64x64x32|
|Max pooling|Stride=2, kernel=2|64x64x32|33x33x32|
|Convolutional|Stride=1, kernel size=3, padding=1|33x33x32|33x33x64|
|Batch normalization|Normalization of output from the convolutional layer|33x33x64|33x33x64|
|Relu|Activation function|33x33x64|33x33x64|
|Max pooling|Stride=2, kernel=2|33x33x64|17x17x64|
|Convolutional|Stride=1, kernel size=3, padding=1|17x17x64|17x17x128|
|Batch normalization|Normalization of output from the convolutional layer|17x17x128|17x17x128|
|Relu|Activation function|17x17x128|17x17x128|
|Max pooling|Stride=2, kernel=2|17x17x128|9x9x128|
|Convolutional|Stride=1, kernel size=3, padding=1|9x9x128|9x9x256|
|Batch normalization|Normalization of output from the convolutional layer|9x9x256|9x9x256|
|Relu|Activation function|9x9x256|9x9x256|
|Max pooling|Stride=2, kernel=2|9x9x256|5x5x256|
|Convolutional|Stride=1, kernel size=3, padding=1|5x5x256|5x5x512|
|Batch normalization|Normalization of output from the convolutional layer|5x5x512|5x5x512|
|Relu|Activation function|5x5x512|5x5x512|
|Max pooling|Stride=2, kernel=2|5x5x512|3x3x512|
|Flatten|'Flatten' the shape into a vector into a linear layer|3x3x512|4608|
|Linear|Linear layer|4608|512|
|Dropout|Probability=0.6|512|512|
|Linear|Output 50 for every class|512|50|


# Optimizer and loss function

Optimizer – Adam (learning rate=0.0001) 

Loss function – Cross-entropy loss

The Adam optimizer was chosen because it dynamically adjusts the learning rate to each parameter during training, so there is no need to adjust the learning rate decay. Of the other optimizers tested (Adagrad and RMSprop), it provided the best results in the test dataset. 

The cross-entropy loss function was chosen because its result can be interpreted as the probability of belonging to each class, which is why it is often used for classification models. 


# Results

The best accuracy achieved is 62%. Some factors that hinder achieving higher accuracy (excluding limits of CNN size and architecture) come from dataset. Images have low resolution and as shown in example images above, sometimes object which is classified is hidden in the background (person holding batteries, which take only small amount of image space, turtle hidden in water, part of towel on a hanger), as well as relatively low amount of images in each class. Considering those factors, 62% accuracy (as opposed to around 2% accuracy for random guessing) seems satisfactory for relatively simple CNN.
