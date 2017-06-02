# Q-net model network.

# Takes as input the screen image resized to 84x84 and
# outputs the number of possible actions
# (18 in atari)

from collections import OrderedDict
from lasagne.layers import InputLayer, DenseLayer, batch_norm
from lasagne.nonlinearities import rectify as ReLU
from lasagne.nonlinearities import linear

try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except ImportError:
    from lasagne.layers import Conv2DLayer as ConvLayer


def build_model(nActions):
    net = OrderedDict()
    net['input'] = InputLayer((None, 4, 84, 84))
    net['conv1'] = batch_norm(ConvLayer(net['input'],
                              num_filters=32,
                              filter_size=8,
                              stride=4,
                              pad='valid',
                              nonlinearity=ReLU))
    net['conv2'] = batch_norm(ConvLayer(net['conv1'],
                              num_filters=64,
                              filter_size=4,
                              stride=2,
                              pad='valid',
                              nonlinearity=ReLU))
    net['conv3'] = batch_norm(ConvLayer(net['conv2'],
                              num_filters=64,
                              filter_size=3,
                              stride=1,
                              pad='valid',
                              nonlinearity=ReLU))
    net['fc4'] = batch_norm(DenseLayer(net['conv3'],
                            num_units=512,
                            nonlinearity=ReLU))
    net['fc5'] = DenseLayer(net['fc4'],
                            num_units=nActions,
                            nonlinearity=linear)
    return net
