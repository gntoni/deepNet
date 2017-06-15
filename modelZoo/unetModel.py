# U-Net implementation
# Paper: https://arxiv.org/abs/1505.04597

from collections import OrderedDict
from lasagne.layers import InputLayer, ConcatLayer, Deconv2DLayer, batch_norm
from lasagne.layers import DimshuffleLayer, ReshapeLayer, NonlinearityLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import rectify as ReLU
from lasagne.nonlinearities import softmax

try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except ImportError:
    from lasagne.layers import Conv2DLayer as ConvLayer


def build_model(nBaseFilters=64,fs1=3,fs2=3):
    net = OrderedDict()
    net['input'] = InputLayer((None, 1, 540, 960))
    net['econv1_1'] = batch_norm(
                                ConvLayer(net['input'],
                                          num_filters=nBaseFilters,
                                          filter_size=fs1,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['econv1_2'] = batch_norm(
                                ConvLayer(net['econv1_1'],
                                          num_filters=nBaseFilters,
                                          filter_size=fs2,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['pool1'] = PoolLayer(net['econv1_2'],
                             pool_size=2,
                             stride=2,
                             ignore_border=False,
                             mode='max')
    net['econv2_1'] = batch_norm(
                                ConvLayer(net['pool1'],
                                          num_filters=nBaseFilters*2,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['econv2_2'] = batch_norm(
                                ConvLayer(net['econv2_1'],
                                          num_filters=nBaseFilters*2,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['pool2'] = PoolLayer(net['econv2_2'],
                             pool_size=2,
                             stride=2,
                             ignore_border=False,
                             mode='max')
    net['econv3_1'] = batch_norm(
                                ConvLayer(net['pool2'],
                                          num_filters=nBaseFilters*4,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['econv3_2'] = batch_norm(
                                ConvLayer(net['econv3_1'],
                                          num_filters=nBaseFilters*4,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['pool3'] = PoolLayer(net['econv3_2'],
                             pool_size=2,
                             stride=2,
                             ignore_border=False,
                             mode='max')
    net['econv4_1'] = batch_norm(
                                ConvLayer(net['pool3'],
                                          num_filters=nBaseFilters*8,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['econv4_2'] = batch_norm(
                                ConvLayer(net['econv4_1'],
                                          num_filters=nBaseFilters*8,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['pool4'] = PoolLayer(net['econv4_2'],
                             pool_size=2,
                             stride=2,
                             ignore_border=False,
                             mode='max')
    net['econv5_1'] = batch_norm(
                                ConvLayer(net['pool4'],
                                          num_filters=nBaseFilters*16,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['econv5_2'] = batch_norm(
                                ConvLayer(net['econv5_1'],
                                          num_filters=nBaseFilters*16,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['upconv1'] = batch_norm(
                                Deconv2DLayer(net['econv5_2'],
                                              num_filters=nBaseFilters*8,
                                              filter_size=2,
                                              stride=2,
                                              crop="valid",
                                              nonlinearity=ReLU))
    net['concat1'] = ConcatLayer(
                            [net['upconv1'], net['econv4_2']],
                            cropping=(None, None, "center", "center"))
    net['dconv1_1'] = batch_norm(
                                ConvLayer(net['concat1'],
                                          num_filters=nBaseFilters*8,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['dconv1_2'] = batch_norm(
                                ConvLayer(net['dconv1_1'],
                                          num_filters=nBaseFilters*8,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['upconv2'] = batch_norm(
                                Deconv2DLayer(net['dconv1_2'],
                                              num_filters=nBaseFilters*4,
                                              filter_size=2,
                                              stride=2,
                                              crop="valid",
                                              nonlinearity=ReLU))
    net['concat2'] = ConcatLayer(
                            [net['upconv2'], net['econv3_2']],
                            cropping=(None, None, "center", "center"))
    net['dconv2_1'] = batch_norm(
                                ConvLayer(net['concat2'],
                                          num_filters=nBaseFilters*4,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['dconv2_2'] = batch_norm(
                                ConvLayer(net['dconv2_1'],
                                          num_filters=nBaseFilters*4,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['upconv3'] = batch_norm(
                                Deconv2DLayer(net['dconv2_2'],
                                              num_filters=nBaseFilters*2,
                                              filter_size=2,
                                              stride=2,
                                              crop="valid",
                                              nonlinearity=ReLU))
    net['concat3'] = ConcatLayer(
                            [net['upconv3'], net['econv2_2']],
                            cropping=(None, None, "center", "center"))
    net['dconv3_1'] = batch_norm(
                                ConvLayer(net['concat3'],
                                          num_filters=nBaseFilters*2,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['dconv3_2'] = batch_norm(
                                ConvLayer(net['dconv3_1'],
                                          num_filters=nBaseFilters*2,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['upconv4'] = batch_norm(
                                Deconv2DLayer(net['dconv3_2'],
                                              num_filters=nBaseFilters,
                                              filter_size=2,
                                              stride=2,
                                              crop="valid",
                                              nonlinearity=ReLU))
    net['concat4'] = ConcatLayer(
                            [net['upconv4'], net['econv1_2']],
                            cropping=(None, None, "center", "center"))
    net['dconv4_1'] = batch_norm(
                                ConvLayer(net['concat4'],
                                          num_filters=nBaseFilters,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['dconv4_2'] = batch_norm(
                                ConvLayer(net['dconv4_1'],
                                          num_filters=nBaseFilters,
                                          filter_size=3,
                                          pad='same',
                                          nonlinearity=ReLU))
    net['output_segmentation'] = ConvLayer(net['dconv4_2'],
                                           num_filters=2,
                                           filter_size=1,
                                           nonlinearity=None)
    net['dimshuffle'] = DimshuffleLayer(net['output_segmentation'],
                                        (1, 0, 2, 3))
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (2, -1))
    net['dimshuffle2'] = DimshuffleLayer(net['reshapeSeg'], (1, 0))
    net['output_flattened'] = NonlinearityLayer(net['dimshuffle2'],
                                                nonlinearity=softmax)
    return net
