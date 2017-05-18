#!/usr/bin/env python

import numpy as np
import cPickle
import lasagne
import theano
import theano.tensor as T

from modelZoo import unetModel
from collections import OrderedDict
from deepNet.core import deepNet
from os import path


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
        self._network = unetModel.build_model()

        if not isinstance(self._network, OrderedDict):
            raise AttributeError("Network model must be an OrderedDict")
        self._inputLayer = self._network[self._network.keys()[0]]
        self._outputLayer = self._network[self._network.keys()[-1]]

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
            self._test_acc = T.mean(T.eq(
                                                T.argmax(
                                                      self._test_prediction,
                                                      axis=1),
                                                self._target_var),
                                    dtype=theano.config.floatX)

            # Compile a function performing a training step on a mini-batch
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
