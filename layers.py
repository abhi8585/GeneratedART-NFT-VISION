#!/usr/bin/env python
"""Provides different layers to be used the model, classes for Image Generation.

This module implement different utilities to standardize the data. 
"""
import tensorflow as tf
from tensor2tensor.layers.common_layers import shape_list, dense

__author__ = "Abhishek Sharma"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Abhishek Sharma"
__email__ = "abhi.sharma1114@gmail.com"
__status__ = "Protoype"

weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None


def flatten(x):
    tensor_shape = shape_list(x)
    return tf.reshape(x, shape=[tensor_shape[0], -1, tensor_shape[-1]])


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, is_training, iteration=1):
    w_shape = shape_list(w)
    w = tf.reshape(w, [-1, w_shape[-1]])  

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(),
                        trainable=False)  

    u_norm = u
    v_norm = None
    for i in range(iteration):
        """power iteration"""
        v_ = tf.matmul(u_norm, w, transpose_b=True)  
        v_norm = l2_norm(v_)

        u_ = tf.matmul(v_norm, w) 
        u_norm = l2_norm(u_)

    
    sigma = tf.matmul(tf.matmul(v_norm, w), u_norm, transpose_b=True) 
    w_norm = w / sigma

    
    with tf.control_dependencies([tf.cond(is_training,
                                          true_fn=lambda: u.assign(u_norm), false_fn=lambda: u.assign(u))]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def batch_norm(x, is_training, scope='batch_norm'):
    return tf.layers.batch_normalization(x,
                                         momentum=0.9,
                                         epsilon=1e-05,
                                         training=is_training,
                                         name=scope)


def spectral_conv2d(x, filters, kernel_size, stride, is_training, padding='SAME', use_bias=True, scope='conv2d'):
    with tf.variable_scope(scope):
        w = tf.get_variable("conv_w",
                            shape=[kernel_size, kernel_size, shape_list(x)[-1], filters],
                            initializer=weight_init,
                            regularizer=weight_regularizer)
        x = tf.nn.conv2d(input=x,
                         filter=spectral_norm(w, is_training),
                         strides=[1, stride, stride, 1],
                         padding=padding)
        if use_bias:
            bias = tf.get_variable("conv_bias", [filters], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)
    return x


def spectral_deconv2d(x, filters, kernel_size, stride, is_training, padding='SAME', use_bias=True, scope='deconv2d'):
    with tf.variable_scope(scope):
        x_shape = shape_list(x)
        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, filters]

        else:
            output_shape = [x_shape[0], x_shape[1] * stride + max(kernel_size - stride, 0),
                            x_shape[2] * stride + max(kernel_size - stride, 0), filters]

        w = tf.get_variable("conv_w",
                            shape=[kernel_size, kernel_size, filters, x_shape[-1]],
                            initializer=weight_init,
                            regularizer=weight_regularizer)
        x = tf.nn.conv2d_transpose(x,
                                   filter=spectral_norm(w, is_training),
                                   output_shape=output_shape,
                                   strides=[1, stride, stride, 1],
                                   padding=padding)
        if use_bias:
            bias = tf.get_variable("conv_bias", [filters], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)

    return x


def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


def attention(x, filters, is_training, scope='attention', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        f = spectral_conv2d(x, filters // 8, kernel_size=1, stride=1, padding='VALID', is_training=is_training,
                            scope='f_conv')  
        g = spectral_conv2d(x, filters // 8, kernel_size=1, stride=1, padding='VALID', is_training=is_training,
                            scope='g_conv')
        h = spectral_conv2d(x, filters, kernel_size=1, stride=1, padding='VALID', is_training=is_training,
                            scope='h_conv')  
        f_flatten = flatten(f) 
        g_flatten = flatten(g)

        s = tf.matmul(g_flatten, f_flatten, transpose_b=True) 
        beta = tf.nn.softmax(s, axis=-1)  

        o = tf.matmul(beta, flatten(h)) 
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=shape_list(x)) 
        x = gamma * o + x

    return x


def residual_block(inputs, output_channel, stride, is_training, scope='residual'):
    with tf.variable_scope(scope):
        x = spectral_conv2d(inputs, output_channel, 3, stride, is_training=is_training, use_bias=False, scope='conv_1')
        x = batch_norm(x, is_training, scope='bn1')
        x = prelu(x)
        x = spectral_conv2d(x, output_channel, 3, stride, is_training=is_training, use_bias=False, scope='conv_2')
        x = batch_norm(x, is_training, scope='bn2')
        x = x + inputs

    return x


def discriminator_block(inputs, output_channel, kernel_size, stride, is_training, scope='d_residual'):
    with tf.variable_scope(scope):
        x = spectral_conv2d(inputs, output_channel, kernel_size, stride, is_training=is_training, use_bias=False,
                            scope='conv_1')
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = spectral_conv2d(x, output_channel, kernel_size, stride, is_training=is_training, use_bias=False,
                            scope='conv_2')
        x = x + inputs

    return x


def prelu(x):
    alphas = tf.get_variable('alpha', shape_list(x)[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.5

    return pos + neg


def PhaseShift(inputs, shape_1, shape_2):
    X = tf.reshape(inputs, shape_1)  
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)



def PixelShuffler(inputs, scale=2):
    size = shape_list(inputs)  
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = size[-1]

    
    channel_target = c // (scale * scale)  
    channel_factor = c // channel_target  

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]  
    shape_2 = [batch_size, h * scale, w * scale, 1]  

    
    input_split = tf.split(inputs, channel_target, axis=3)  
    output = tf.concat([PhaseShift(x, shape_1, shape_2) for x in input_split], axis=3)  

    return output



def ResBlockUp(inputs, output_channel, is_training, scope='residual', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        x = batch_norm(inputs, is_training, scope='bn1')
        x = tf.nn.leaky_relu(x)
        x = spectral_deconv2d(x, output_channel, 3, stride=2, is_training=is_training, scope='deconv1')
        x = batch_norm(x, is_training, scope='bn2')
        x = tf.nn.leaky_relu(x)
        x = spectral_conv2d(x, output_channel, 3, stride=1, is_training=is_training, scope='conv1')

        
        skip = spectral_deconv2d(inputs, output_channel, 3, stride=2, is_training=is_training, scope='deconv_skip')
        x = x + skip

    return x


def ResBlockDown(inputs, output_channel, is_training, scope='residual', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        x = batch_norm(inputs, is_training, scope='bn1')
        x = tf.nn.leaky_relu(x)
        x = spectral_conv2d(x, output_channel, 3, stride=2, is_training=is_training, scope='conv1')
        x = batch_norm(x, is_training, scope='bn2')
        x = tf.nn.leaky_relu(x)
        x = spectral_conv2d(x, output_channel, 3, stride=1, is_training=is_training, scope='conv2')

        
        skip = spectral_conv2d(inputs, output_channel, 3, stride=2, is_training=is_training, scope='conv_skip')
        x = x + skip

    return x
