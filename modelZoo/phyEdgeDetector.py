#!/usr/bin/env python

from collections import OrderedDict
from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import dropout, batch_norm
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import rectify as ReLU
from lasagne.nonlinearities import softmax

try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except ImportError:
    print "[Warning] Could not use DNN. Loading regular Conv Layer"
    from lasagne.layers import Conv2DLayer as ConvLayer


def build_model(timesteps, pX, pY):
    net = OrderedDict()
    net['input'] = InputLayer(shape=(None, timesteps, pX, pY))
    net['conv1_1'] = batch_norm(
                            ConvLayer(
                                 net['input'],
                                 num_filters=32,
                                 filter_size=(3, 3),
                                 nonlinearity=ReLU,
                                 pad='same'))

    net['conv1_2'] = batch_norm(
                            ConvLayer(
                                 net['conv1_1'],
                                 num_filters=32,
                                 filter_size=(3, 3),
                                 nonlinearity=ReLU,
                                 pad='same'))

    net['pool1'] = PoolLayer(net['conv1_2'], pool_size=(2, 2))

    net['conv2_1'] = batch_norm(
                            ConvLayer(
                                 net['pool1'],
                                 num_filters=64,
                                 filter_size=(3, 3),
                                 nonlinearity=ReLU,
                                 pad='same'))

    net['conv2_2'] = batch_norm(
                            ConvLayer(
                                 net['conv2_1'],
                                 num_filters=64,
                                 filter_size=(3, 3),
                                 nonlinearity=ReLU,
                                 pad='same'))

    net['pool2'] = PoolLayer(net['conv2_2'], pool_size=(2, 2))

    net['conv3_1'] = batch_norm(
                            ConvLayer(
                                 net['pool2'],
                                 num_filters=128,
                                 filter_size=(3, 3),
                                 nonlinearity=ReLU,
                                 pad='same'))

    net['conv3_2'] = batch_norm(
                            ConvLayer(
                                 net['conv3_1'],
                                 num_filters=128,
                                 filter_size=(3, 3),
                                 nonlinearity=ReLU,
                                 pad='same'))

    net['pool3'] = PoolLayer(net['conv3_2'], pool_size=(2, 2))

    net['fcLayer1'] = batch_norm(
                            DenseLayer(
                                 dropout(net['pool3'], p=0.5),
                                 num_units=512,
                                 nonlinearity=ReLU))

    net['output'] = DenseLayer(
                                 dropout(net['fcLayer1'], p=0.5),
                                 num_units=2,
                                 nonlinearity=softmax)
    return net
