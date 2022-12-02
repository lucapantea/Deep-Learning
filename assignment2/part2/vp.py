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

"""Defines various kinds of visual-prompting modules for images."""
import torch
import torch.nn as nn
import numpy as np

from utils import DummyArgs


class PadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        self.device = args.device

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Dimensions image: (batch, channels, height, width)
        self.pad_left = nn.Parameter(data=torch.rand((1, 3, image_size-2*pad_size, pad_size)), requires_grad=True)
        self.pad_right = nn.Parameter(data=torch.rand((1, 3, image_size-2*pad_size, pad_size)), requires_grad=True)
        self.pad_up = nn.Parameter(data=torch.rand((1, 3, pad_size, image_size)), requires_grad=True)
        self.pad_down = nn.Parameter(data=torch.rand((1, 3, pad_size, image_size)), requires_grad=True)

        # Defining prompt
        base = torch.zeros((1, 3, image_size-2*pad_size, image_size-2*pad_size)).to(self.device)
        self._prompt = torch.cat([self.pad_left.to(self.device).data, base, self.pad_right.to(self.device).data], dim=3)
        self._prompt = torch.cat([self.pad_up.to(self.device).data, self._prompt, self.pad_down.to(self.device).data], dim=2)
        self._prompt = self._prompt.to(self.device)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = x.to(self.device)
        x += self._prompt
        return x
        #######################
        # END OF YOUR CODE    #
        #######################


class FixedPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For refernece, this prompt should look like Fig 2(a) in the PDF.
    """
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.device = args.device
        self.prompt_size = args.prompt_size
        self.patch = nn.Parameter(data=torch.rand((1, 3, self.prompt_size, self.prompt_size)), requires_grad=True)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = x.to(self.device)
        x[:, :, :self.prompt_size, :self.prompt_size] += self.patch.to(self.device).data
        return x
        #######################
        # END OF YOUR CODE    #
        #######################


class RandomPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a random patch in the image.
    For refernece, this prompt should look like Fig 2(b) in the PDF.
    """
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.device = args.device
        self.image_size = args.image_size
        self.prompt_size = args.prompt_size
        self.patch = nn.Parameter(data=torch.rand((1, 3, self.prompt_size, self.prompt_size)), requires_grad=True)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = x.to(self.device)
        random_pos_x = np.random.randint(low=self.prompt_size, high=self.image_size)
        random_pos_y = np.random.randint(low=self.prompt_size, high=self.image_size)
        x[:, :, random_pos_x-self.prompt_size:random_pos_x, random_pos_y-self.prompt_size:random_pos_y] += self.patch.to(self.device).data
        return x
        #######################
        # END OF YOUR CODE    #
        #######################

class CheckerBoardPrompter(nn.Module):
    """
    Defines visual-prompt as a checker-board patch in the image.
    This helps to achieve near-random performance.
    """
    def __init__(self, args):
        super(CheckerBoardPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        self.device = args.device
        self.image_size = args.image_size
        self.prompt_size = args.prompt_size
        pattern = self.get_checkerboard_pattern((self.image_size, self.image_size), self.prompt_size)
        # Important: use CheckerBoardPrompter.prompt.grad *= mask after calling backward()
        self.mask = torch.from_numpy(pattern).to(self.device)
        self.data = torch.rand((1, 3, self.image_size, self.image_size)).to(self.device)
        self.prompt = nn.Parameter(data=self.data * self.mask, requires_grad=True)

    def forward(self, x):
        x = x.to(self.device)
        x += self.prompt.to(self.device).data
        return x
    
    @staticmethod
    def get_checkerboard_pattern(image_size, tile_size):
        return (np.indices(image_size) // tile_size).sum(axis=0) % 2

