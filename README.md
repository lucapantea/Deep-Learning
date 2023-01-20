# Deep Learning 1 Assignments at UvA, edition 2022

This repository contains the practical assignments of the Deep Learning 1 course at the University of Amsterdam.

*Course website: [https://uvadlc.github.io/](https://uvadlc.github.io/)*\
*Course edition: Fall 2022*\
*Course taught by: Dr. Yuki M. Asano*

## Assignment 1 - MLPs and Backpropagation
In the first assignment, the equations of backpropagation for basic modules in a vanilla neural network are derived from scratch,
and then implemented as modules in order to create a simple multi-layer perceptron (MLP). The dataset used for this assignment is CIFAR-10.
Model accuracies, hyperparameters, and other statistics can be automatically generated using ```compare_models.py```.

## Assignment 2 - Transfer Learning for CNNs, Visual Prompting, Graph Neural Networks
The assignment is organized in two parts. The first part covers transfer learning for CNNs, 
where you fine-tune an existing network to adapt it to a new dataset. 
In the second part, you prompt CLIP to perform image classification, 
both in the zero-shot setting and by learning a visual prompt. 
The corresponding results can be found under the ```results``` directory.

##### Deliverables for Transfer Learning with CNNs
* Compare popular CNN architectures on ImageNet using data from the [PyTorch website](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights).
* Use the official PyTorch implementations to compute and compare the inference speed, memory usage, and parameter count for the same models.
* Adapt a ResNet-18 model from ImageNet to CIFAR-100 by resetting its last layer.
* Perform augmentations to improve performance.

##### Deliverables for Visual Prompting
* Evaluate CLIP-B/32 in the zero-shot setting on CIFAR-10 and CIFAR-100 (`clipzs.py`).
* Prompt CLIP on two new downstream tasks by changing the text template (`--prompt`) and class labels (`--class_names`). You can visualize your predictions with `--visualize_predictions`.
* Learn visual prompts for the CIFAR-10 and CIFAR-100 datasets (`learner.py`, `vpt_model.py`, `vp.py`).
* Experiment with the prompt design to get near-random performance on CIFAR-100.
* Evaluate the robustness of the learnt prompts to distributional shifts (`robustness.py`).
* Evaluate each dataset's learnt prompt on the concatenation of both CIFAR-10 and CIFAR-100 (`cross_dataset.py`).


## Assignment 3 - Deep Generative Models
The final assignment contains 2 parts. Part 1 is to implement in pytorch a variational autoencoder (VAE), and part 2 an adversarial autoencoder (AAE). 

For the VAE part, we will train the model on generating 4-bit MNIST images. The original MNIST dataset contains images with pixel values between 0 and 1. To discretize those, we multiply pixel values with 16 and map the result to the closest integer value (rounding down 16 to 15). This is a 4-bit representation of the original image. Standard RGB images are usually using 8-bit encodings (i.e. values between 0 and 255), but to simplify the task, we only use 4 bits here.

For the AAE part, We will train the model on generating MNIST images.
