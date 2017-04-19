#!/usr/bin/env python

"""
Utility functions

Author: Toni Gabas.  a.gabas@aist.go.jp
"""

from __future__ import print_function

import numpy as np
import theano
import h5py


"""
Helper function to get minibatches over the full batch.
"""


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Helper function to iteratively return minibatches  of a given
    size from a batch.

    Parameters
    ----------
    inputs : int
        Array containing the dataset of inputs
    targets : int
        Array containing the dataset of outputs
    batchsize: int
        Length of the minibatch
    shuffle: bool
        Define if the samples in the minibatch are taken
        consecutively or randomly

    Yields
    -------
    array
        Minibatch of the inputs batch
    array
        Minibatch of the targets batch

    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def load_data(inputsPath, labelsPath=None):
    if inputsPath[-3:] == "npy" and labelsPath[-3:] == "npy":
            patches = np.load(inputsPath)
            labels = np.load(labelsPath)
            N = len(patches)
            patchX, patchY = patches[0].shape
            split_indices = (int(N*0.8), int((N*0.8)+(N*0.1)))
            patches = patches.reshape(
                (-1, 1, patchX, patchY))
            labels = labels.astype('int32')
            X_train, X_val, X_test = np.split(patches, split_indices)
            y_train, y_val, y_test = np.split(labels, split_indices)
            return X_train, y_train, X_val, y_val, X_test, y_test
    elif inputsPath[-2:] == "h5" and labelsPath is None:
            with h5py.File(inputsPath, 'r') as hf:
                X_train = np.array(hf.get('patches_train')).astype(theano.config.floatX)
                N, px, py = X_train.shape
                X_train = X_train.reshape((-1, 1, px, py))
                X_val = np.array(
                    hf.get('patches_valid')).reshape((-1, 1, px, py)).astype(theano.config.floatX)
                X_test = np.array(
                    hf.get('patches_test')).reshape((-1, 1, px, py)).astype(theano.config.floatX)
                y_train = np.array(hf.get('labels_train')).astype('int32')
                y_val = np.array(hf.get('labels_valid')).astype('int32')
                y_test = np.array(hf.get('labels_test')).astype('int32')
            return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        raise ValueError("ERROR: Training dataset fileformat incorrect")
