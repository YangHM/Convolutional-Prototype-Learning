#################################################
import numpy as np
import tensorflow as tf
import cPickle as pickle
import struct

##################################################

# compute distances between the sample features
# with the centers
def distance(features, centers):
    f_2 = tf.reduce_sum(tf.pow(features, 2), axis=1, keep_dims=True)
    c_2 = tf.reduce_sum(tf.pow(centers, 2), axis=1, keep_dims=True)
    dist = f_2 - 2*tf.matmul(features, centers, transpose_b=True) + tf.transpose(c_2, perm=[1,0])

    return dist

# the cross entorpy loss for the traditional 
# softmax layer based  neural networks
def softmax_loss(logits, labels):
    labels = tf.to_int32(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
        logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

# L2 regular loss
def regular_loss(name):
    params = tf.get_collection(name)
    return tf.add_n([tf.nn.l2_loss(i) for i in params])

# margin based classification loss (MCL)
def mcl_loss(features, labels, centers, margin):
    dist = distance(features, centers)

    values, indexes = tf.nn.top_k(-dist, k=2, sorted=True)
    top2 = -values
    d_1 = top2[:, 0]
    d_2 = top2[:, 1]

    row_idx = tf.range(tf.shape(labels)[0], dtype=tf.int32)
    idx = tf.stack([row_idx, labels], axis=1)
    d_y = tf.gather_nd(dist, idx, name='dy')

    indicator = tf.cast(tf.nn.in_top_k(-dist, labels, k=1), tf.float32)
    d_c = indicator*d_2 + (1-indicator)*d_1

    loss = tf.nn.relu(d_y-d_c+margin, name='loss')
    mean_loss = tf.reduce_mean(loss, name='mean_loss')

    return mean_loss

# generalized margin based classification loss (GMCL)
def gmcl_loss(features, labels, centers, margin):
    dist = distance(features, centers)

    values, indexes = tf.nn.top_k(-dist, k=2, sorted=True)
    top2 = -values
    d_1 = top2[:, 0]
    d_2 = top2[:, 1]

    row_idx = tf.range(tf.shape(labels)[0], dtype=tf.int32)
    idx = tf.stack([row_idx, labels], axis=1)
    d_y = tf.gather_nd(dist, idx, name='dy')

    indicator = tf.cast(tf.nn.in_top_k(-dist, labels, k=1), tf.float32)
    d_c = indicator*d_2 + (1-indicator)*d_1

    loss = tf.nn.relu((d_y-d_c)/(d_y+d_c)+margin, name='loss')
    mean_loss = tf.reduce_mean(loss, name='mean_loss')

    return mean_loss

# minimum classification error loss (MCE)
def mce_loss(features, labels, centers, epsilon):
    dist = distance(features, centers)

    values, indexes = tf.nn.top_k(-dist, k=2, sorted=True)
    top2 = -values
    d_1 = top2[:, 0]
    d_2 = top2[:, 1]

    row_idx = tf.range(tf.shape(labels)[0], dtype=tf.int32)
    idx = tf.stack([row_idx, labels], axis=1)
    d_y = tf.gather_nd(dist, idx, name='dy')

    indicator = tf.cast(tf.nn.in_top_k(-dist, labels, k=1), tf.float32)
    d_c = indicator*d_2 + (1-indicator)*d_1

    measure = d_y - d_c

    loss = tf.sigmoid(epsilon*measure, name='loss')
    mean_loss = tf.reduce_mean(loss, name='mean_loss')

    return mean_loss

# distance based cross entropy loss (DCE)
def dce_loss(features, labels, centers, T):
    dist = distance(features, centers)
    logits = -dist / T

    mean_loss = softmax_loss(logits, labels)

    return mean_loss

# prototype loss (PL)
def pl_loss(features, labels, centers):
    batch_num = tf.cast(tf.shape(features)[0], tf.float32)
    batch_centers = tf.gather(centers, labels)
    dis = features - batch_centers
    return tf.div(tf.nn.l2_loss(dis), batch_num)
    
##################################################

# return the training operation to train the network
def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op
    
##################################################

# evaluation operation in traditional softmax-layer based NNs
def base_evaluation(logits, labels):
    prediction = tf.argmax(logits, axis=1, name='prediction')
    correct = tf.equal(tf.cast(prediction, tf.int32), labels, name='correct')
    return tf.reduce_sum(tf.cast(correct, tf.float32), name='evaluation')

# prediction operation in CPL or GCPL framwork
def predict(features, centers):
    dist = distance(features, centers) 
    prediction = tf.argmin(dist, axis=1, name='prediction')
    return tf.cast(prediction, tf.int32)

# evaluation operation in CPL or GCPL framework
def evaluation(features, labels, centers):
    dist = distance(features, centers) 

    prediction = tf.argmin(dist, axis=1, name='prediction')
    correct = tf.equal(tf.cast(prediction, tf.int32), labels, name='correct')
    return tf.reduce_sum(tf.cast(correct, tf.float32), name='evaluation')


##################################################

# construct prototypes (centers) for each class
def construct_center(features, num_classes):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0))

    return centers

# operations used to initialize the prototypes in
# each class (with the mean vector of the class)
def init_centers(features, labels, centers, counts):
    add_op = tf.scatter_add(centers, labels, features, name='add_op')
    
    unique_label, unique_index, unique_count = tf.unique_with_counts(labels)
    count_op = tf.scatter_add(counts, unique_label, unique_count, name='count_op')

    average_op = tf.assign(centers, centers/tf.cast(tf.reshape(counts, [-1,1]), tf.float32),
        name='average_op')

    return add_op, count_op, average_op



