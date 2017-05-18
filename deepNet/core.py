#!/usr/bin/env python

"""
Base class for Neural Network generation with
Lasagne and Theano

Author: Toni Gabas.  a.gabas@aist.go.jp
"""

from __future__ import print_function
from os import path
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from collections import OrderedDict

import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import xmltodict

from modelZoo import cifar10Net
from utils import iterate_minibatches


class deepNet(object):
    """
    Base class to generate deep neural networks
    
    Attributes:
        trainParams (TYPE): Description
    """
    __metaclass__ = ABCMeta

    def __init__(self, config):
            """Summary
            """
            # self._input_var = T.tensor4('inputs', dtype=theano.config.floatX)
            self._target_var = T.ivector('targets')

            self.trainParams = {
                'learning_rate': None,
                'momentum': None,
                'batch_size': None,
                'num_epochs': None,
                'output_models_folder': "",
                'save_freq': None  # save model every N epochs
            }
            self.setModel()
            self.loadTrainParams(config)
            self.setTrainFuncs()

    @abstractmethod
    def setModel(self):
        self._network = None
        
        if not isinstance(self._network, OrderedDict):
            raise AttributeError("Network model must be an OrderedDict")
        self._inputLayer = self._network[self._network.keys()[0]]
        self._outputLayer = self._network[self._network.keys()[-1]]

    def loadTrainParams(self, configFile):
            """
            Function that loads a xml configuration file, parses it and
            returns a filled "trainParams" dictionary variable.
            
            Returns:
                Dict: Dictionary containing the training parameters loaded fro
                        the file.
            
            Args:
                configFile (str): Path to the XML file containing the
                        training configuration parameters.
            """
            with open(configFile) as fd:
                f = xmltodict.parse(fd.read())
            if "trainConfig" not in f:
                raise IOError("No training data found in the xml file.")
            if "learning_rate" in f["trainConfig"]:
                self.trainParams["learning_rate"] = float(
                                        f["trainConfig"]["learning_rate"])
                print("Loaded new value for learning_rate: {}"
                      .format(f["trainConfig"]["learning_rate"]))
            if "momentum" in f["trainConfig"]:
                self.trainParams["momentum"] = float(
                                        f["trainConfig"]["momentum"])
                print("Loaded new value for momentum: {}"
                      .format(f["trainConfig"]["momentum"]))
            if "batch_size" in f["trainConfig"]:
                self.trainParams["batch_size"] = int(
                                        f["trainConfig"]["batch_size"])
                print("Loaded new value for batch_size: {}"
                      .format(f["trainConfig"]["batch_size"]))
            if "num_epochs" in f["trainConfig"]:
                self.trainParams["num_epochs"] = int(
                                        f["trainConfig"]["num_epochs"])
                print("Loaded new value for num_epochs: {}"
                      .format(f["trainConfig"]["num_epochs"]))
            if "output_models_folder" in f["trainConfig"]:
                self.trainParams["output_models_folder"] = \
                                f["trainConfig"]["output_models_folder"]
                print("Loaded new value for output_models_folder: {}"
                      .format(f["trainConfig"]["output_models_folder"]))
            if "save_freq" in f["trainConfig"]:
                self.trainParams["save_freq"] = int(
                                        f["trainConfig"]["save_freq"])
                print("Loaded new value for save_freq: {}"
                      .format(f["trainConfig"]["save_freq"]))

    @abstractmethod
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
            self._loss = None

            # Create update expressions for training, 
            # i.e., how to modify the
            # parameters at each training step.
            # (relative to the net params)
            # see lasagne.updates
            self._params = lasagne.layers.get_all_params(
                                                    self._outputLayer,
                                                    trainable=True)
            self._updates = None

            # Create a loss expression for validation/testing. The difference
            # here is that we do a deterministic forward pass through the
            # network, disabling dropout layers.
            self._test_prediction = lasagne.layers.get_output(
                                            self._outputLayer,
                                            deterministic=True)
            self._test_loss = None
            # Also create an expression for the classification accuracy:
            self._test_acc = None

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

    def get_network_params(self):
        """
        Gets the current weights of all the layers in the network.
        
        Returns:
            ndarray: weights array containing as many dimmensions
                    as layers in the current network.
        """
        return lasagne.layers.get_all_param_values(self._outputLayer)

    def set_network_params(self, model):
        """
        Sets the weights of all the layers in the network.
        
        Args:
            model (ndarray): weights array containing as many
                    dimmensions as layers in the network.
        """
        lasagne.layers.set_all_param_values(self._network, model)
        return

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

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Training function.  For each epoch all the minibatches in the
        train data will be used for gradient descent. After each epoch
        is completed the loss and accuracy over the validation set will
        be computed.
        Finally after the given number of epochs have been completed,
        the loss and accuracy over the test set will be evaluated.
        The parameters of the model will be saved periodically.
        
        Args:
            X_train (ndarray): Input array with the training data
            y_train (ndarray): Labels array for the training data
            X_val (ndarray): Input array with the validation data
            y_val (ndarray): Labels array for the validation data
            X_test (ndarray): Input array with the test data
            y_test (ndarray): Labels array for the test data
        
        """
        print("Starting training...")
        # We iterate over epochs:
        num_epochs = self.trainParams["num_epochs"]
        batchsize = self.trainParams["batch_size"]
        for epoch in range(num_epochs):
            start_time = time.time()
            # In each epoch, we do a full pass over the training data:
            train_err, _ = self._run_epoch(
                                                X_train,
                                                y_train,
                                                batchsize,
                                                training=True
                                                )
            val_err, val_acc = self._run_epoch(
                                                    X_val,
                                                    y_val,
                                                    batchsize,
                                                    training=False
                                                    )

            # Print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err))
            print("  validation loss:\t\t{:.6f}".format(val_err))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc))

            # Save the model after a number of epochs
            if epoch % self.trainParams["save_freq"] == 0:
                    params = lasagne.layers.get_all_param_values(
                        self._outputLayer)
                    modelname = "model_" + str(epoch) + "_epoch"
                    np.save(
                            path.join(
                                    self.trainParams["output_models_folder"],
                                    modelname),
                            params)

        # After training, we compute and print the test error:
        test_err, test_acc = self._run_epoch(
                                                    X_test,
                                                    y_test,
                                                    batchsize,
                                                    training=False
                                                    )
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err))
        print("  test accuracy:\t\t{:.2f} %".format(test_acc))
        params = lasagne.layers.get_all_param_values(self._outputLayer)
        np.save("model", params)

    def test(self, X, y):
        """
        Function that, given a pair of inputs and labels, will split them into
        minibatches and run them through the network returning the average
        error and accuracy.
        
        Args:
            X (ndarray): Input data.
            y (ndarray): Input labels.
        
        Returns:
            (float, float): Average Error and Average Accuracy
        """
        test_err, test_acc = self._run_epoch(
                                                X,
                                                y,
                                                self.trainParams["batch_size"],
                                                training=False
                                                )
        print("Test results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err))
        print("  test accuracy:\t\t{:.2f} %".format(test_acc))
        return (test_err, test_acc)

    def run(self, X):
        """
        Function that takes one input and runs it through the network.
        The output is the direct predictions of the network.
        
        Args:
            X (ndarray): Input data
        
        Returns:
            ndarray: Network output
        """
        return self._pred_fn(X)
