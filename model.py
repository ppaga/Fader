import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import Conv2dLayer, DeConv2dLayer, DenseLayer
from tensorlayer.layers import ConcatLayer, InputLayer, BatchNormLayer
from tensorlayer.layers import FlattenLayer, DropoutLayer, ReshapeLayer
from tensorlayer.layers import LambdaLayer
from keras.layers import Conv2DTranspose
#from tensorlayer.layers import LambdaLayer as Lambda
#from tensorlayer.layers import DeConv2dLayerLayer as DeConv2dLayer
#from tensorlayer.layers import BatchNormLayer as BatchNorm
#from tensorlayer.layers import InputLayer as Input
from tensorlayer.activation import leaky_relu

from skimage.transform import resize
from matplotlib.pyplot import imshow, imread
import matplotlib.pyplot as plt

kernel_size = [4]*2
strides = (1,2,2,1)
padding = 'SAME'
alpha = 0.2
p_dropout = 0.3
c_dim = 3

init_filter_enc = 16
init_filter_dec = 512
init_filter_disc = 512

def C_enc(net, shape, name):
    layername = name+'conv'
    net = Conv2dLayer(net, shape=shape, strides = strides, padding = padding, name=layername)
    layername = name+'batchnorm'
    net = BatchNormLayer(net, name=layername)
    net.outputs = leaky_relu(net.outputs,alpha=alpha)
    return net
    
def Encoder(net, name):
    layername = name+'/input'
    net = InputLayer(net, name = layername)
    
    filters = init_filter_enc
    shape = kernel_size+[c_dim,filters]
    
    for i in range(4):
        layername = name+'/block'+str(i)+'/'
        net = C_enc(net, shape, layername)
        shape[2] = filters
        if i<5:
            filters = 2*filters
        shape[3]=filters
    outputs = Conv2dLayer(net, shape = shape, strides = strides, padding = padding)
    return outputs

def keras_deconv(x, filters_out, name):
    deconv_fn = lambda x: Conv2DTranspose(filters_out, kernel_size = kernel_size, strides = strides[1:-1], padding=padding)(x)
    net = LambdaLayer(x, deconv_fn, name=name)
    return net


def C_dec(net, y, filters_out, name, activation = tf.nn.relu):
    layername = name+'lambda'
    y = LambdaLayer(y, lambda x: tf.tile(x,(1,2,2,1)), name=layername)
    layername = name+'concat'
    net = ConcatLayer([net, y], concat_dim=3, name=layername)
    layername = name+'deconv'
    net = keras_deconv(net, filters_out, layername)
    layername = name+'batchnorm'
    net = BatchNormLayer(net, name=layername)
    return net, y

def Decoder(net, y, name):
    layername = name+'/'+'attribute_input'
    y = InputLayer(y, name=layername)

    filters_out = init_filter_dec
    for i in range(4):
        layername = name+'/block'+str(i)+'/'                
        net, y = C_dec(net, y, filters_out, layername)
        filters_out = filters_out//2
    filters_out = 3
    outputs = keras_deconv(net, filters_out, name+'output_deconv')
    return outputs
    
def Discriminator(net, n, name):
    filters_in = net.outputs.shape.as_list()[-1]
#    layername = name+'/input'
#    net = InputLayer(net, name = layername)
    
    filters_out = init_filter_disc
    shape = kernel_size + [filters_in, filters_out]
    layername = name+'/conv'
    net = Conv2dLayer(net, shape = shape, strides = strides, 
                          padding = padding, act = tf.nn.relu, name=layername)
    net = FlattenLayer(net)

    n_units = 512
    layername = name+'/dense_0'
    net = DenseLayer(net, n_units = n_units, act = tf.nn.relu, name=layername)
    layername = name+'/dropout_0'
    net = DropoutLayer(net, keep = 1-p_dropout, name = layername)
    n_units = n
    layername = name+'/dense_1'
    net = DenseLayer(net, n_units = 2*n_units, name = layername)
    layername = name+'/dropout_1'
    net = DropoutLayer(net, keep = 1-p_dropout, name = layername)
    logits = ReshapeLayer(net, (-1, n_units, 2))
    return logits
