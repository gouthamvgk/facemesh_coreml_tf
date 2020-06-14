import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
print(tf.__version__)

def res_block(inp_tensor, out_filters, kernel_size=(3,3), strides=(1,1), data_format="channels_first"):
    if data_format == "channels_first":
        shared_axes = [2,3]
    else:
        shared_axes = [1,2]
    depth1 = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding="SAME",data_format=data_format)(inp_tensor)
    conv1 = Conv2D(out_filters, kernel_size=(1,1), strides=(1,1),data_format=data_format)(depth1)
    add1_out = Add()([inp_tensor, conv1])
    act_1 = PReLU(shared_axes=shared_axes)(add1_out)
    depth2 = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding="SAME", data_format=data_format)(act_1)
    conv2 = Conv2D(out_filters, kernel_size=(1,1), strides=(1,1), data_format=data_format)(depth2)
    add2_out = Add()([act_1, conv2])
    act2 = PReLU(shared_axes=shared_axes)(add2_out)
    return act2

def down_sampling(inp_tensor, out_filters, strides=(2,2), kernel_size=(3,3), data_format="channels_first"):
    
    if data_format == "channels_first":
        pad_dim = out_filters - inp_tensor.shape[1]
        shared_axes = [2,3]
        pad_val = [[0,0], [0, pad_dim], [0,0], [0,0]]
    else:
        pad_dim = out_filters - inp_tensor.shape[3]
        shared_axes = [1,2]
        pad_val = [[0,0], [0,0], [0,0], [0, pad_dim]]
    
    depth1 = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding="SAME", data_format=data_format)(inp_tensor)
    conv1 = Conv2D(out_filters, kernel_size=(1,1), strides=(1,1), data_format=data_format)(depth1)
    max_pool = MaxPool2D(pool_size=(2,2), strides=strides, data_format=data_format)(inp_tensor)
    if pad_dim != 0:
        max_pool = tf.pad(max_pool, pad_val)
    add1_out = Add()([conv1, max_pool])
    act_1 = PReLU(shared_axes=shared_axes)(add1_out)
    return act_1

