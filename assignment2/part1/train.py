################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False

    # Randomly initialize and modify the model's last layer for CIFAR100.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    torch.nn.init.normal_(model.fc.weight.data, 0, 0.01)
    torch.nn.init.zeros_(model.fc.bias.data)
    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    cifar100_train, cifar100_val = get_train_validation_set(data_dir)
    train_loader = data.DataLoader(dataset=cifar100_train, batch_size=batch_size, shuffle=True,
                                   drop_last=True)
    val_loader = data.DataLoader(dataset=cifar100_val, batch_size=batch_size, shuffle=False,
                                 drop_last=False)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(params=model.fc.parameters(), lr=lr)

    # Initialize the criterion
    loss_module = nn.CrossEntropyLoss()

    # Saving best (valid acc) model for the test set
    best_valid_acc = 0.0

    # Training loop with validation after each epoch. Save the best model.
    print('Beginning Training...')
    for epoch in range(epochs):
        model.train()
        train_total, train_correct = 0, 0
        for step, train_data in enumerate(train_loader, 0):
            inputs, targets = train_data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Perform forward pass
            preds = model(inputs)

            # Calculate training loss & acc
            loss = loss_module(preds, targets)
            train_correct += (preds.argmax(1) == targets).sum().item()
            train_total += targets.shape[0]

            # Perform backprop & update step
            loss.backward()
            optimizer.step()

        val_accuracy = evaluate_model(model, val_loader, device)
        print(f'[Epoch {epoch + 1}] Validation accuracy: {val_accuracy:.3f}')

        # Saving model with the best validation accuracy
        if val_accuracy > best_valid_acc:
            best_valid_acc = val_accuracy
            torch.save(model.state_dict(), checkpoint_name)

    # Load the best model on val accuracy and return it.
    model.load_state_dict(torch.load(checkpoint_name))
    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    print('Beginning evaluation...')
    model.eval()
    val_total, val_correct = 0, 0

    # Loop over the dataset and compute the accuracy. Return the accuracy
    for step, val_data in enumerate(data_loader, 0):
        inputs, targets = val_data
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            # Perform forward pass
            preds = model(inputs)

            # Calculate evaluation loss & accuracy
            val_correct += (preds.argmax(1) == targets).sum().item()
            val_total += targets.shape[0]

    accuracy = val_correct / val_total
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    MODEL_NAME = 'modified_resnet.pth'
    CHECKPOINT_PATH = '../saved_models/part1'

    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = get_model()

    # Get the augmentation to use
    pass

    # Train the model
    best_model = train_model(model, lr, batch_size, epochs, data_dir,
                             checkpoint_name=CHECKPOINT_PATH + '/' + MODEL_NAME,
                             device=device, augmentation_name=augmentation_name)

    # Initialize the test data loader
    cifar100_test = get_test_set(data_dir)
    test_loader = data.DataLoader(dataset=cifar100_test, batch_size=batch_size, shuffle=False,
                                  drop_last=False)
    # Evaluate the model on the test set
    test_accuracy = evaluate_model(best_model, test_loader, device)
    print(f'Accuracy of ResNet18 on the test set: {test_accuracy:.3f}')
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
