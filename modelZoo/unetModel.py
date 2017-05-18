# U-Net implementation
# Paper: https://arxiv.org/abs/1505.04597

from collections import OrderedDict
from lasagne.layers import InputLayer, ConcatLayer, Deconv2DLayer
from lasagne.layers import DimshuffleLayer, ReshapeLayer, NonlinearityLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import rectify as ReLU
from lasagne.nonlinearities import softmax

try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except ImportError:
    from lasagne.layers import Conv2DLayer as ConvLayer


def build_model():
    net = OrderedDict()
    net['input'] = InputLayer((None, 1, 540, 960))
    net['econv1_1'] = ConvLayer(net['input'],
                                num_filters=64,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['econv1_2'] = ConvLayer(net['econv1_1'],
                                num_filters=64,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['pool1'] = PoolLayer(net['econv1_2'],
                             pool_size=2,
                             stride=2,
                             ignore_border=False,
                             mode='max')
    net['econv2_1'] = ConvLayer(net['pool1'],
                                num_filters=128,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['econv2_2'] = ConvLayer(net['econv2_1'],
                                num_filters=128,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['pool2'] = PoolLayer(net['econv2_2'],
                             pool_size=2,
                             stride=2,
                             ignore_border=False,
                             mode='max')
    net['econv3_1'] = ConvLayer(net['pool2'],
                                num_filters=256,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['econv3_2'] = ConvLayer(net['econv3_1'],
                                num_filters=256,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['pool3'] = PoolLayer(net['econv3_2'],
                             pool_size=2,
                             stride=2,
                             ignore_border=False,
                             mode='max')
    net['econv4_1'] = ConvLayer(net['pool3'],
                                num_filters=512,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['econv4_2'] = ConvLayer(net['econv4_1'],
                                num_filters=512,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['pool4'] = PoolLayer(net['econv4_2'],
                             pool_size=2,
                             stride=2,
                             ignore_border=False,
                             mode='max')
    net['econv5_1'] = ConvLayer(net['pool4'],
                                num_filters=1024,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['econv5_2'] = ConvLayer(net['econv5_1'],
                                num_filters=1024,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['upconv1'] = Deconv2DLayer(net['econv5_2'],
                                   num_filters=512,
                                   filter_size=2,
                                   stride=2,
                                   crop="valid",
                                   nonlinearity=ReLU)
    net['concat1'] = ConcatLayer(
                            [net['upconv1'], net['econv4_2']],
                            cropping=(None, None, "center", "center"))
    net['dconv1_1'] = ConvLayer(net['concat1'],
                                num_filters=512,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['dconv1_2'] = ConvLayer(net['dconv1_1'],
                                num_filters=512,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['upconv2'] = Deconv2DLayer(net['dconv1_2'],
                                   num_filters=256,
                                   filter_size=2,
                                   stride=2,
                                   crop="valid",
                                   nonlinearity=ReLU)
    net['concat2'] = ConcatLayer(
                            [net['upconv2'], net['econv3_2']],
                            cropping=(None, None, "center", "center"))
    net['dconv2_1'] = ConvLayer(net['concat2'],
                                num_filters=256,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['dconv2_2'] = ConvLayer(net['dconv2_1'],
                                num_filters=256,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['upconv3'] = Deconv2DLayer(net['dconv2_2'],
                                   num_filters=128,
                                   filter_size=2,
                                   stride=2,
                                   crop="valid",
                                   nonlinearity=ReLU)
    net['concat3'] = ConcatLayer(
                            [net['upconv3'], net['econv2_2']],
                            cropping=(None, None, "center", "center"))
    net['dconv3_1'] = ConvLayer(net['concat3'],
                                num_filters=128,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['dconv3_2'] = ConvLayer(net['dconv3_1'],
                                num_filters=128,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['upconv4'] = Deconv2DLayer(net['dconv3_2'],
                                   num_filters=64,
                                   filter_size=2,
                                   stride=2,
                                   crop="valid",
                                   nonlinearity=ReLU)
    net['concat4'] = ConcatLayer(
                            [net['upconv4'], net['econv1_2']],
                            cropping=(None, None, "center", "center"))
    net['dconv4_1'] = ConvLayer(net['concat4'],
                                num_filters=64,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
    net['dconv4_2'] = ConvLayer(net['dconv4_1'],
                                num_filters=64,
                                filter_size=3,
                                pad='same',
                                nonlinearity=ReLU)
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
