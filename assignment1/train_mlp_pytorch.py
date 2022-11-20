################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy

from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false
    negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_classes = 10
    conf_mat = np.zeros(shape=(n_classes, n_classes), dtype=np.float32)
    # iterate over batch targets
    for batch, target in enumerate(targets):
        conf_mat[target][predictions[batch].argmax()] += 1
    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_classes = confusion_matrix.shape[0]
    metrics = {
        'accuracy': np.trace(confusion_matrix) / np.sum(confusion_matrix),
        'precision': np.array([confusion_matrix[n_class, n_class] / np.sum(confusion_matrix, axis=0)[n_class]
                               for n_class in range(n_classes)]),
        'recall': np.array([confusion_matrix[n_class, n_class] / np.sum(confusion_matrix, axis=1)[n_class]
                            for n_class in range(n_classes)])
    }
    metrics['f1_beta'] = (1 + beta ** 2) * np.multiply(metrics['precision'], metrics['recall']) / \
                         (beta ** 2 * metrics['precision'] + metrics['recall'])
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()
    conf_mat = np.zeros(shape=(num_classes, num_classes), dtype=float)
    print("Beginning testing...")
    for step, data in enumerate(data_loader, 0):
        inputs, targets = data

        with torch.no_grad():
            # Perform forward pass
            preds = model(inputs)
            batch_conf_mat = confusion_matrix(preds, targets)
            conf_mat += batch_conf_mat

    metrics = confusion_matrix_to_metrics(conf_mat)
    print()
    print(f'Confusion matrix for the testset: \n{conf_mat}\n')
    print(f'Metrics for testset calculated from Confusion Matrix: {metrics}')
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Initialize model and criterion
    model = MLP(n_inputs=32 * 32 * 3, n_hidden=hidden_dims, n_classes=10, use_batch_norm=use_batch_norm)
    loss_module = nn.CrossEntropyLoss()

    # Initialize optimizer
    optimizer = optim.SGD(params=model.parameters(), lr=lr)

    # Saving best (valid acc) model for the test set
    best_model = None
    best_valid_acc = 0.0

    # Logging info - training loop
    val_accuracies = []
    log_freq = 10  # logging max 50 points for each training iteration
    logging_info = {'train_loss': [], 'train_acc': [], 'valid_loss': []}

    print("Beginning Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for step, data in enumerate(cifar10_loader.get('train'), 0):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Perform forward pass
            preds = model(inputs)

            # Calculate training loss & acc
            loss = loss_module(preds, targets)
            train_loss += loss.item()
            train_correct += (preds.argmax(1) == targets).sum().item()
            train_total += targets.shape[0]

            # Perform backprop & SGD update step
            loss.backward()
            optimizer.step()

            if step % log_freq == log_freq - 1:
                print(f'[Epoch {epoch + 1}, Step {step + 1:5d}] '
                      f'Train loss: {train_loss / log_freq:.3f}, '
                      f'Train acc: {train_correct / train_total:.4f}')
                logging_info['train_loss'].append(round(train_loss / log_freq, 3))
                logging_info['train_acc'].append(round(train_correct / train_total, 3))
                train_loss = 0.0

        # Validation loop
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        for step, data in enumerate(cifar10_loader.get('validation'), 0):
            inputs, targets = data
            with torch.no_grad():
                # Perform forward pass
                preds = model(inputs)

                # Calculate evaluation loss & accuracy
                loss = loss_module(preds, targets)
                valid_loss += loss.item()
                valid_correct += (preds.argmax(1) == targets).sum().item()
                valid_total += targets.shape[0]

        valid_acc = valid_correct / valid_total
        val_accuracies.append(valid_acc)
        logging_info['valid_loss'].append(round(valid_loss / len(cifar10.get("validation")), 3))
        print(f'Validation loss: {valid_loss / len(cifar10.get("validation")):.3f}, '
              f'Validation acc: {valid_acc:.4f}')

        # Saving model with the best validation accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = deepcopy(model)

    # Evaluate model with best one obtained in training
    metrics = evaluate_model(best_model, cifar10_loader.get('test'))
    test_accuracy = metrics.get('accuracy')
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)
    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    print(test_accuracy)
    # Feel free to add any additional functions, such as plotting of the loss curve here

    def learning_rates_experiment():
        # Learning rates experimenting and plotting
        start_lr = 0.000001
        lrs = []
        for _ in range(9):
            lrs.append(start_lr)
            start_lr *= 10
        print(f'Learning rates: {lrs}')

        # figure 1: accuracy v learning rates
        fig1, ax0 = plt.subplots(1, 1, tight_layout=True, figsize=(12, 4))

        # figure 2: loss function v training steps (for each LR)
        fig2, ((ax1, ax2, ax3),
               (ax4, ax5, ax6),
               (ax7, ax8, ax9)) = plt.subplots(3, 3, tight_layout=True, figsize=(20, 14))

        best_valid_accuracies = []
        for ax, lr in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9], lrs):
            kwargs['lr'] = lr
            model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
            best_valid_accuracies.append(max(val_accuracies))

            steps_training = range(len(logging_dict.get('train_loss')))
            ax.plot(steps_training, logging_dict.get('train_loss'))
            ax.set_xlabel('Steps')
            ax.set_ylabel('Training Loss')
            ax.set_title(f'Lr: %.E' % lr)
            ax.set_ylim([1, 4])

        string_lrs = list(map(lambda _lr: '%.E' % _lr, lrs))
        ax0.plot(string_lrs[:len(best_valid_accuracies)], best_valid_accuracies, "-o")
        ax0.set_xlabel('Learning Rates')
        ax0.set_ylabel('Best Validation Accuracy')
        min_val, max_val = min(best_valid_accuracies), max(best_valid_accuracies)
        ax0.plot(np.argmin(best_valid_accuracies), min_val, "s", label=f"Min accuracy: {min_val:.4f}")
        ax0.plot(np.argmax(best_valid_accuracies), max_val, "D", label=f"Max accuracy: {max_val:.4f}")
        ax0.set_ylim([0, 0.5])

        ax0.legend(loc='upper right')
        plt.show()

    def plot_metrics(val_accuracies, logging_dict):
        # Plot Validation accuracy over steps and training loss over epoch & steps
        fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(12, 4))
        steps_training = range(len(logging_dict.get('train_loss')))
        epochs = [epoch + 1 for epoch in range(len(val_accuracies))]
        ax1.plot(steps_training, logging_dict.get('train_loss'))
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Training Loss')
        ax1.set_ylim([1, 3])

        ax2.plot(epochs, val_accuracies, "-o")
        min_val, max_val = min(val_accuracies), max(val_accuracies)
        ax2.plot(np.argmin(val_accuracies) + 1, min_val, "s", label=f"Min accuracy: {min_val:.4f}")
        ax2.plot(np.argmax(val_accuracies) + 1, max_val, "D", label=f"Max accuracy: {max_val:.4f}")
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_xticks(epochs)

        plt.show()

    # todo: delete
    # # import libraries
    # import pandas as pd
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # # Create an array with the colors you want to use
    # colors = ["#69b3a2", "#4374B3"]
    # sns.set_palette(sns.color_palette(colors))
    # data = {'precision': [0.55894591, 0.61947791, 0.29565217, 0.34598214, 0.39710611, 0.38923557, 0.52287582,
    # 0.59749145, 0.52429668, 0.61633663],
    #         'recall': [0.403, 0.617, 0.408, 0.155, 0.494, 0.499, 0.56, 0.524, 0.615, 0.498],
    #         'Classes': [epoch + 1 for epoch in range(10)]}
    #
    # data = {'0.1': [0.55681259, 0.61945328, 0.29646043, 0.34181223, 0.39787879, 0.39008514, 0.52321924,
    #                 0.59666291, 0.5250634, 0.61488998],
    #         '1': [0.46833236, 0.61823647, 0.34285714, 0.2140884, 0.4402852, 0.43733567, 0.54080155,
    #               0.55833777, 0.56603774, 0.55088496],
    #         '10': [0.40411632, 0.61702444, 0.4064707, 0.15585178, 0.49280945, 0.49761063,
    #                0.55960661, 0.52463892, 0.61394839, 0.4989485],
    #         'Classes': [epoch + 1 for epoch in range(10)]}
    #
    # # df = pd.melt(pd.DataFrame(data=data), id_vars='Classes', var_name='Metric', value_name='Value')
    # # sns.catplot(x='Classes', y='Value', hue='Metric', data=df, kind='bar')
    #
    # df = pd.melt(pd.DataFrame(data=data), id_vars='Classes', var_name='Beta value', value_name='F1_beta score')
    # sns.catplot(x='Classes', y='F1_beta score', hue='Beta value', data=df, kind='bar')
    # plt.show()
