################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    z = torch.randn_like(std) * std + mean
    #######################
    # END OF YOUR CODE    #
    #######################
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    KL = torch.exp(torch.mul(2, log_std)) + torch.pow(mean, 2) - torch.ones_like(mean) - torch.mul(2, log_std)
    KLD = torch.mul(0.5, torch.sum(KL, dim=-1))
    #######################
    # END OF YOUR CODE    #
    #######################
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    bpd = elbo * np.log2(np.e) * 1/torch.prod(torch.tensor(img_shape[1:]))
    #######################
    # END OF YOUR CODE    #
    #######################
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Range for the percentiles
    grid_lower_bound = 0.5/grid_size
    grid_upper_bound = (grid_size-0.5)/grid_size
    percentiles = torch.linspace(start=grid_lower_bound, end=grid_upper_bound, steps=grid_size)

    # Normal icdf for z value at percentile
    normal = torch.distributions.Normal(loc=0, scale=1)
    icdf = normal.icdf(percentiles)
    z = torch.cartesian_prod(icdf, icdf)  # Equivalent to meshgrid, yet output is tensor

    # Sampling procedure
    x = decoder(z).softmax(1)  # x shape: [grid_size**2, 16, 28, 28]
    shape_x = x.shape
    x = torch.flatten(x.permute(0, 2, 3, 1), end_dim=-2)  # x shape: [grid_size**2 * 28 * 28, 16]: [313600, 16]
    samples = torch.multinomial(x, num_samples=1)  # x_samples shape: [313600, 1]
    samples = samples.reshape(shape_x[0], shape_x[2], shape_x[3]).unsqueeze(1)  # x_samples shape: [grid_size**2, 28, 28]
    samples = samples.float() / 15  # Converting 4-bit images to values between 0 and 1

    # Generate grid
    img_grid = make_grid(samples, nrow=grid_size, normalize=True, value_range=(0, 1), pad_value=0.5)
    img_grid = img_grid.detach().cpu()
    #######################
    # END OF YOUR CODE    #
    #######################

    return img_grid
