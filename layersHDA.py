import numpy as np
import tensorflow as tf


class TFHDA:
    '''
    
    Class for a tensorflow high dimensional algebraic object
        - Designed to interface with 
        - Current implementation is only for 'Complex' and 'SplitComplex'
    
    '''
    def __init__(self, real, imag, HDAtype = 'Complex'):
        self.r = real
        self.i = imag
        self.HDAtype = HDAtype

def typeClone(x, HDAtype):
    '''
    
    Clones input 'x' to be correct high-dimensional algebra type
    
    '''

    xout = TFHDA(x, tf.zeros_like(x), HDAtype = HDAtype)
    return xout

def conv2d(x, weightShape = None, biasDim = None, convStride = None, convPadding = None, mu = 0, sigma = 0.1):
    '''
    2D convolutional layer overloaded for complex arithmetic, implements via weight-sharing and
        works as wrapper file for tf.nn.conv2d()
    
    Inputs:
        x - input array
        weightShape - tuple with shape 
        biasDim - size of bias, should match last dimension of weightShape
        convStride - array of strides for the convolution
        convPadding - padding time for convolution
        mu - mean for weight initialization
        sigma - std dev for weight initialization
    
    Output:
        convOut - convolutional layer output
    
    '''
    
    
    if type(x) is TFHDA:
        conv_W_r = tf.Variable(tf.truncated_normal(shape=weightShape, mean = mu, stddev = sigma))
        conv_W_i = tf.Variable(tf.truncated_normal(shape=weightShape, mean = mu, stddev = sigma))
        conv_b_r = tf.Variable(tf.zeros(biasDim))
        conv_b_i = tf.Variable(tf.zeros(biasDim))
        
        if x.HDAtype == 'Complex':
            convOutr = tf.nn.conv2d(x.r, conv_W_r, strides=convStride, padding=convPadding) - \
            tf.nn.conv2d(x.i, conv_W_i, strides=convStride, padding=convPadding) + conv_b_r
            
            convOuti = tf.nn.conv2d(x.i, conv_W_r, strides=convStride, padding=convPadding) + \
            tf.nn.conv2d(x.r, conv_W_i, strides=convStride, padding=convPadding) + conv_b_i
            
            convOut = TFHDA(convOutr, convOuti)
            
        elif x.HDAtype == 'SplitComplex':
            convOutr = tf.nn.conv2d(x.r, conv_W_r, strides=convStride, padding=convPadding) + \
            tf.nn.conv2d(x.i, conv_W_i, strides=convStride, padding=convPadding) + conv_b_r
            convOuti = tf.nn.conv2d(x.i, conv_W_r, strides=convStride, padding=convPadding) + \
            tf.nn.conv2d(x.r, conv_W_i, strides=convStride, padding=convPadding) + conv_b_i
            convOut = TFHDA(convOutr, convOuti)
        
    else:
        conv_W = tf.Variable(tf.truncated_normal(shape=weightShape, mean = mu, stddev = sigma))
        conv_b = tf.Variable(tf.zeros(biasDim))
        convOut = tf.nn.conv2d(x, conv_W, strides=convStride, padding=convPadding) + conv_b
    
    return convOut

def affine(x, weightShape = None, biasDim = None, mu = 0, sigma = 0.1):
    if type(x) is TFHDA:
        fc_W_r = tf.Variable(tf.truncated_normal(shape=weightShape, mean = mu, stddev = sigma))
        fc_W_i = tf.Variable(tf.truncated_normal(shape=weightShape, mean = mu, stddev = sigma))
        fc_b_r = tf.Variable(tf.zeros(biasDim))
        fc_b_i = tf.Variable(tf.zeros(biasDim))
        
        if x.HDAtype == 'Complex':
            affineOutr = tf.matmul(x.r, fc_W_r) - tf.matmul(x.i, fc_W_i) + fc_b_r
            affineOuti = tf.matmul(x.r, fc_W_i) + tf.matmul(x.i, fc_W_r) + fc_b_i
            
            affineOut = TFHDA(affineOutr, affineOuti)
            
        elif x.HDAtype == 'SplitComplex':
            affineOutr = tf.matmul(x.r, fc_W_r) + tf.matmul(x.i, fc_W_i) + fc_b_r
            affineOuti = tf.matmul(x.r, fc_W_i) + tf.matmul(x.i, fc_W_r) + fc_b_i
            
            affineOut = TFHDA(affineOutr, affineOuti)
        
    else:
        fc_W = tf.Variable(tf.truncated_normal(shape=weightShape, mean = mu, stddev = sigma))
        fc_b = tf.Variable(tf.zeros(biasDim))
        affineOut = tf.matmul(x, fc_W) + fc_b
    
    return affineOut


def avgpool(x, sizes = None, strides = None, padding = None):
    '''
    
    Performs average pooling. If input x is 
    
    '''
    
    if type(x) is TFHDA:
        mags = magnitude(x)
        out_r = tf.nn.avg_pool(x.r, ksize=sizes, strides=strides, padding=padding)
        out_i = tf.nn.avg_pool(x.i, ksize=sizes, strides=strides, padding=padding)
        out = TFHDA(out_r, out_i)
        
    else:
        out = tf.nn.avg_pool(x, ksize=sizes, strides=strides, padding=padding)
        
    return out

def relu(x):
    '''
    
    Implements ReLU function as defined in the paper
    
    '''
    
    
    if type(x) is TFHDA:
        mask = tf.cast(tf.greater_equal(x.r, tf.zeros_like(x.r)), tf.float32)
        out = TFHDA(tf.multiply(mask,x.r), tf.multiply(mask,x.i))
    else:
        out = tf.nn.relu(x)
    
    return out


def flatten(x):
    '''
    
    Flattens input array along each dimension if input is high-dimensional algebraic valued
    
    '''
    
    if type(x) is TFHDA:
        out = TFHDA(tf.contrib.layers.flatten(x.r), tf.contrib.layers.flatten(x.i))
    else:
        out = tf.contrib.layers.flatten(x)
    
    return out

def magnitude(x):
    '''
    Compute magnitude of input if input is HDA object. 
        - If input is real-valued, pass through
    
    '''
    
    if type(x) is TFHDA:
        out = tf.sqrt(tf.square(x.r) + tf.square(x.i))
    else:
        out = x
    
    return out


