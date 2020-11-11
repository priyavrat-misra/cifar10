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
>    - [with data augmentation](https://github.com/priyavrat-misra/cifar10/blob/master/train_with_validation.ipynb "train_with_aug.ipynb")
> * [Evaluating the model's results and making cool graphs!](https://github.com/priyavrat-misra/cifar10/blob/master/results.ipynb "results.ipynb")

## Results:
> || without data augmentation | with data augmentation |
> | :- | -: | -: |
> | Test Accuracy | 76.68% | 79.76% |

<br>

__Note:__ this project uses Tensorboard as an evaluation utility for checking running losses, accuracies, histograms and so on. So if you are wondering why there are no outputs (running loss, epoch number etc) while the network is training, use Tensorboard (_open terminal, change path to project's repo and run this command `tensorboard --logdir=runs`_) or if you aren't a fan of 3rd party tools, adding a bunch of print statements will do the job.
