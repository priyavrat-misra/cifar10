# CIFAR10 Classification with PyTorch

![Cover](https://github.com/priyavrat-misra/cifar10/blob/master/visualizations/test_results_with_aug.png?raw=true "sample test results visualization")

This project uses the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) for training. It consists of `60000` `32x32` colour images in 10 classes, with `6000` images per class. There are `50000` training images and `10000` test images. <br>
The dataset is divided into five training batches and one test batch, each with `10000` images. The test batch contains exactly `1000` randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly `5000` images from each class.
(_Source:https://www.cs.toronto.edu/~kriz/cifar.html_)

<br>

## Steps:
> * [Exploring the dataset](https://github.com/priyavrat-misra/cifar10/blob/master/data_exploration.ipynb "data_exploration.ipynb")
> * [Defining a neural network architecture](https://github.com/priyavrat-misra/cifar10/blob/master/network.py "network.py")
> * Hyper-parameter search and training the model
>    - [without data augmentation](https://github.com/priyavrat-misra/cifar10/blob/master/train.ipynb "train.ipynb")
>    - [with data augmentation](https://github.com/priyavrat-misra/cifar10/blob/master/train_with_aug.ipynb "train_with_aug.ipynb")
>    - [with VGG16 (transfer learning)](https://github.com/priyavrat-misra/cifar10/blob/master/train_with_vgg16.ipynb "train_with_vgg16.ipynb")
> * [Evaluating the model's results and making cool graphs!](https://github.com/priyavrat-misra/cifar10/blob/master/results.ipynb "results.ipynb")

## Results:
> || Train Accuracy | Test Accuracy |
> | :- | -: | -: |
> | without data augmentation | *81.69% | 76.68% |
> | with data augmentation | 85.15% | 79.76% |
> | with transfer learning (VGG-16) | 92.89% | 85.93% |


_<sub>* - running accuracy</sub>_

<br>