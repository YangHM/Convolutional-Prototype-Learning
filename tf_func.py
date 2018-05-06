#################################################
import numpy as np
import tensorflow as tf
import cPickle as pickle
import time
import struct

#################################################

# activation function: ReLU
def ReLU(x):
	return tf.nn.relu(x)

# activation function: Leaky ReLU
class LReLU(object):
	def __init__(self, leak=1.0/3):
		self.leak = leak
	
	def act(self, x):
		f1 = 0.5 * (1+self.leak)
		f2 = 0.5 * (1-self.leak)
		return f1*x + f2*tf.abs(x)

# activation function: PReLU
def PReLU(inputs, name='prelu'):
	with tf.name_scope(name):
		leaky = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), 
			name='leaky')
		f1 = 0.5 * (1 + leaky)
		f2 = 0.5 * (1 - leaky)
		return f1*inputs + f2*tf.abs(inputs)

# operations in a normal convolutional layer
def Conv(inputs, ksize, strides=[1,1,1,1], padding='SAME', activation=None, regular=False, bias=True, name='conv'):
	with tf.name_scope(name):
		scale = np.sqrt(6./(np.prod(ksize[0:3]) + np.prod(ksize[0:2])*ksize[-1]))
		weights = tf.Variable(tf.random_uniform(ksize, -scale, scale),
			name='weights')
		biases = tf.Variable(tf.zeros([ksize[-1]]),
			name='biases')
		if bias:
			conv_out = tf.nn.conv2d(inputs, weights, strides=strides, padding=padding) + biases
		else:
			conv_out = tf.nn.conv2d(inputs, weights, strides=strides, padding=padding)
		if regular:
			tf.add_to_collection('regular', weights)
		return conv_out if activation==None else activation(conv_out)

# the max pooling operation
def Max_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'):
	return tf.nn.max_pool(inputs, ksize, strides, padding)

# operatins in a normal fully-connected layer
def FC(inputs, n_in, n_out, activation=None, regular=False, name='fc'):
	with tf.name_scope(name):
		scale = np.sqrt(6./(n_in+n_out))
		weights = tf.Variable(tf.random_uniform([n_in, n_out], -scale, scale),
			name='weights')
		biases = tf.Variable(tf.zeros([n_out]),
			name='biases')
		if regular:
			tf.add_to_collection('regular', weights)
		lin_out = tf.matmul(inputs, weights) + biases
		return lin_out if activation==None else activation(lin_out)

# drop-out operation
def Dropout(inputs, keep_prob):
	return tf.nn.dropout(inputs, keep_prob)

# padding operation
def Pad(inputs, size, mode='CONSTANT'):
	return tf.pad(inputs, size, mode)

# batch normalization for the convolutional layer
def BN_Conv(input, n_out, phase_train, decay=0.9, name='bn_conv'):
	with tf.name_scope(name):
		beta = tf.Variable(tf.zeros([n_out]), dtype=tf.float32, name='beta')
		gama = tf.Variable(tf.ones([n_out]), dtype=tf.float32, name='gama')
		ema_mean = tf.Variable(tf.zeros([n_out]), dtype=tf.float32, name='ema_mean')
		ema_var = tf.Variable(tf.ones([n_out]), dtype=tf.float32, name='ema_var')

		batch_mean, batch_var = tf.nn.moments(input, [0,1,2])
		ema_mean_op = tf.assign(ema_mean, ema_mean*decay+batch_mean*(1-decay))
		ema_var_op = tf.assign(ema_var, ema_var*decay+batch_var*(1-decay))

		def mean_var_with_update():
			with tf.control_dependencies([ema_mean_op, ema_var_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(phase_train, mean_var_with_update,
			lambda: (ema_mean, ema_var))

		return tf.nn.batch_normalization(input, mean, var, beta, gama, 1e-5)

# batch normalization for the fully-connected layer
def BN_Full(input, n_out, phase_train, decay=0.9, name='bn_full'):
	with tf.name_scope(name):
		beta = tf.Variable(tf.zeros([n_out]), dtype=tf.float32, name='beta')
		gama = tf.Variable(tf.ones([n_out]), dtype=tf.float32, name='gama')
		ema_mean = tf.Variable(tf.zeros([n_out]), dtype=tf.float32, name='ema_mean')
		ema_var = tf.Variable(tf.ones([n_out]), dtype=tf.float32, name='ema_var')

		batch_mean, batch_var = tf.nn.moments(input, [0])
		ema_mean_op = tf.assign(ema_mean, ema_mean*decay+batch_mean*(1-decay))
		ema_var_op = tf.assign(ema_var, ema_var*decay+batch_var*(1-decay))

		def mean_var_with_update():
			with tf.control_dependencies([ema_mean_op, ema_var_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(phase_train, mean_var_with_update,
			lambda: (ema_mean, ema_var))

		return tf.nn.batch_normalization(input, mean, var, beta, gama, 1e-5)


