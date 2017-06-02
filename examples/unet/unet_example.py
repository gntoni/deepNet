#!/usr/bin/env python

import numpy as np
import cPickle
import lasagne
import theano
import theano.tensor as T

from os import path
from modelZoo import unetModel
from collections import OrderedDict
from deepNet.core import deepNet
from deepNet.utils import iterate_minibatches
from tqdm import tqdm


def load_data(datapath, n_batches):
    data = []
    labels = []
    for n in range(n_batches):
        with open(path.join(datapath, 'data_batch_'+str(n+1)), "rb") as fo:
            batch = cPickle.load(fo)
            data.append(batch["data"])
            labels.append(batch["labels"])
    return (np.concatenate(data), np.concatenate(labels))


class unet(deepNet):
    def __init__(self, config):
        deepNet.__init__(self, config)

    def setModel(self):
        self._target_var = T.dmatrix('targets')
        self._network = unetModel.build_model(nBaseFilters=10)

        if not isinstance(self._network, OrderedDict):
            raise AttributeError("Network model must be an OrderedDict")
        self._inputLayer = self._network[self._network.keys()[0]]
        self._outputLayer = self._network[self._network.keys()[-1]]

    def _run_epoch(self, X, y, batchsize, training=False):
        """
        Function that takes a pair of input data and labels, splits i them
        into minibatches and pass them through the network.
        If training (training = True), parameters of the network will be
        updated.
        
        Args:
            X (ndarray): Input data
            y (ndarray): Labels
            batchsize (TYPE): Size of the desired minibatches
            training (bool, optional): If true, updates of the network
                    parameters with Stochastic Gradient descend will be
                    performed after each iteration.
        
        Returns:
            (float, float): Average Error and Average Accuracy
                    When training only error is returned (Accuracy = None)
        """
        err = 0
        acc = 0
        batches = 0
        for batch in tqdm(iterate_minibatches(
                                              X,
                                              y,
                                              batchsize,
                                              shuffle=training),
                          total=len(X)/batchsize):
            inputs, targets = batch
            targets = targets.swapaxes(1, 0)
            targets = targets.reshape((2, -1))
            targets = targets.swapaxes(0, 1)
            inputs = np.asarray(inputs, dtype=theano.config.floatX)
            targets = np.asarray(targets, dtype=theano.config.floatX)

            if training:
                err += self._train_fn(inputs, targets)
            else:
                verr, vacc = self._val_fn(inputs, targets)
                err += verr
                acc += vacc
            batches += 1
        if training:
            return (err/batches, None)
        else:
            return (err/batches, (acc/batches)*100)

    def setTrainFuncs(self):
            """
            Set of functions to perform the traininig/validation of the data.
            This function must be overridden in inherited classes to modify
            the training behaviour

            """
            # Create a loss expression for training,
            # i.e., a scalar objective
            # (relative to the net output)
            # we want to minimize (e.g. cross-entropy loss):
            # Could add some weight decay as well here,
            # see lasagne.regularization.
            self._prediction = lasagne.layers.get_output(self._outputLayer)
            self._loss = lasagne.objectives.categorical_crossentropy(
                                        self._prediction,
                                        self._target_var
                                        )
            self._loss = self._loss.mean()

            # Create update expressions for training,
            # i.e., how to modify the
            # parameters at each training step.
            # (relative to the net params)
            # see lasagne.updates
            self._params = lasagne.layers.get_all_params(
                                                    self._outputLayer,
                                                    trainable=True)
            self._updates = lasagne.updates.adamax(
                            self._loss,
                            self._params,
                            learning_rate=self.trainParams["learning_rate"])

            # Create a loss expression for validation/testing. The difference
            # here is that we do a deterministic forward pass through the
            # network, disabling dropout layers.
            self._test_prediction = lasagne.layers.get_output(
                                            self._outputLayer,
                                            deterministic=True)
            self._test_loss = lasagne.objectives.categorical_crossentropy(
                                            self._test_prediction,
                                            self._target_var)
            self._test_loss = self._test_loss.mean()

            # Also create an expression for the classification accuracy:
            self._test_acc = lasagne.objectives.categorical_accuracy(
                                            self._test_prediction,
                                            self._target_var)
            self._test_acc = self._test_acc.mean()

            # (by giving the updates dictionary) and returning
            # the corresponding training loss:
            self._train_fn = theano.function(
                [self._inputLayer.input_var, self._target_var],
                self._loss,
                updates=self._updates)

            # Compile a second function computing the validation loss
            # and accuracy:
            self._val_fn = theano.function(
                [self._inputLayer.input_var, self._target_var],
                [self._test_loss, self._test_acc])

            # Compile one last function returning the prediction for
            # testing without labels available.
            self._pred_fn = theano.function(
                [self._inputLayer.input_var],
                self._test_prediction)
            return
