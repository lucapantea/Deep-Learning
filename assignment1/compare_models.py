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
# Date Created: 2022-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch
from matplotlib import pyplot as plt
from train_mlp_pytorch import train

import torch
import torch.nn as nn
import torch.optim as optim
# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    kwargs = {'hidden_dims': None, 'use_batch_norm': False, 'lr': 0.1,
              'batch_size': 128, 'epochs': None, 'seed': 42, 'data_dir': 'data/'}
    epochs = 20
    hidden_dims = [512, 256, 128]
    results = {'val_accuracies': [], 'test_accuracy': [],
               'logging_dict': []}
    for i in range(len(hidden_dims)):
        n_hidden = hidden_dims[-(i + 1):]
        kwargs['hidden_dims'] = n_hidden
        kwargs['epochs'] = epochs
        _, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
        results.get('val_accuracies').append(val_accuracies)
        results.get('test_accuracy').append(test_accuracy)
        results.get('logging_dict').append(logging_dict)
        with open(results_filename, 'w') as _file:
            results_json = json.dumps(results)
            _file.write(results_json)
    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    with open(results_filename, 'r') as _file:
        results = json.load(_file)
        val_accuracies_array, test_accuracy_array, logging_dict_array = (results.get(key) for key in results.keys())

    # ax1: train loss, ax2: train acc, ax3: validation acc
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True, figsize=(16, 5))

    for i in range(len(val_accuracies_array)):
        val_accuracies = val_accuracies_array[i]
        test_accuracy = test_accuracy_array[i]
        logging_dict = logging_dict_array[i]
        epochs = len(val_accuracies)
        train_steps = range(len(logging_dict.get('train_loss')))
        ax1.plot(train_steps, logging_dict.get('train_loss'),
                 label=f'Model {i + 1}, Test accuracy: {test_accuracy}')
        ax2.plot(train_steps, logging_dict.get('train_acc'), label=f'Model {i + 1}, Test accuracy: {test_accuracy}')
        ax3.plot(range(1, epochs + 1), val_accuracies, label=f'Model {i + 1}, Test accuracy: {test_accuracy}')

        ax1.set_xlabel('Steps')
        ax1.set_ylim([0, 3])
        ax1.set_ylabel('Training Loss')
        ax2.set_xlabel('Steps')
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Training Accuracy')
        ax3.set_xlabel('Epochs')
        ax3.set_xticks(range(1, epochs + 1))
        ax3.set_ylabel('Validation Accuracy')

    # Legend business
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    # Put a legend below current axis
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
               fancybox=True, shadow=True, ncol=1, prop={'size': 13})
    plt.show()
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'results.json'
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)
