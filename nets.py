#################################################
from tf_func import *

##################################################

# net used for the MNIST dataset, also appeared in paper:
# "A discriminative feature learning approach for deep
# face recognition"
def mnist_net(images):
    inputs = tf.transpose(images, perm=[0,2,3,1])
    
    conv1 = Conv(inputs, [5,5,1,32], activation=ReLU)

    conv2 = Conv(conv1, [5,5,32,32], activation=ReLU)

    pool1 = Max_pool(conv2, padding='VALID')

    conv3 = Conv(pool1, [5,5,32,64], activation=ReLU)

    conv4 = Conv(conv3, [5,5,64,64], activation=ReLU)

    pool2 = Max_pool(conv4, padding='VALID')

    conv5 = Conv(pool2, [5,5,64,128], activation=ReLU)

    conv6 = Conv(conv5, [5,5,128,128], activation=ReLU)

    pool3 = Max_pool(conv6, padding='VALID')

    fc1 = FC(tf.reshape(pool3, [-1, 3*3*128]), 3*3*128, 2)
    fc1_out = ReLU(fc1)

    logits = FC(fc1_out, 2, 10)

    return fc1, logits

##################################################

# net used for the CIFAR dataset, also appeared in paper:
# "Striving for simplicity: The all convolutional net"
def cifar_net1(images, keep_rates):
    inputs = Dropout(tf.transpose(images, perm=[0, 2, 3, 1]), keep_rates[0])

    conv1 = Conv(inputs, [3,3,3,96], activation=ReLU, regular=True)
    conv2 = Conv(conv1, [3,3,96,96], activation=ReLU, regular=True)
    pool1 = Max_pool(conv2, ksize=[1,3,3,1])
    drop1 = Dropout(pool1, keep_rates[1])

    conv3 = Conv(drop1, [3,3,96,192], activation=ReLU, regular=True)
    conv4 = Conv(conv3, [3,3,192,192], activation=ReLU, regular=True)
    pool2 = Max_pool(conv4, ksize=[1,3,3,1])
    drop2 = Dropout(pool2, keep_rates[2])

    conv5 = Conv(drop2, [3,3,192,192], activation=ReLU, regular=True)
    conv6 = Conv(conv5, [1,1,192,192], activation=ReLU, regular=True)
    conv7 = Conv(conv6, [1,1,192,10], activation=ReLU, regular=True)

    pool3 = tf.nn.avg_pool(conv7, [1,8,8,1], [1,8,8,1], 'SAME')

    flatten = tf.reshape(pool3, [-1, 10])

    logits = FC(flatten, 10, 10, regular=True)

    return flatten, logits

##################################################

# net used for OLHWDB dataset, also appeared in paper:
#"Online and offline handwritten Chinese character 
#recognition: A comprehensive study and new benchmark"
# but we add BN layers
def subnet_a(inputs, ksize, phase_train, bn_decay, activation, keep_rate):
    conv = Conv(inputs, ksize)
    bn = BN_Conv(conv, ksize[-1], phase_train, bn_decay)
    act = activation(bn)
    conv_out = Dropout(act, keep_rate)
    return conv_out

def subnet_b(inputs, n_in, n_out, phase_train, bn_decay, activation, keep_rate):
    fc = FC(inputs, n_in, n_out)
    bn = BN_Full(fc, n_out, phase_train, bn_decay)
    act = activation(bn)
    fc_out = Dropout(act, keep_rate)
    return fc_out

def character_net(direct_map, keep_rates, leaky, phase_train, bn_decay):
    leaky_relu = LReLU(leaky)
    func = leaky_relu.act

    inputs = tf.transpose(direct_map, perm=[0,2,3,1])
    
    conv1 = subnet_a(inputs, [3,3,8,50], phase_train, bn_decay, func, keep_rates[0])

    conv2 = subnet_a(conv1, [3,3,50,100], phase_train, bn_decay, func, keep_rates[1])

    pool1 = Max_pool(conv2, padding='VALID')
    
    conv3 = subnet_a(pool1, [3,3,100,150], phase_train, bn_decay, func, keep_rates[2])

    conv4 = subnet_a(conv3, [3,3,150,200], phase_train, bn_decay, func, keep_rates[3])
    
    pool2 = Max_pool(conv4, padding='VALID')

    conv5 = subnet_a(pool2, [3,3,200,250], phase_train, bn_decay, func, keep_rates[4])

    conv6 = subnet_a(conv5, [3,3,250,300], phase_train, bn_decay, func, keep_rates[5])

    pool3 = Max_pool(conv6, padding='VALID')

    conv7 = subnet_a(pool3, [3,3,300,350], phase_train, bn_decay, func, keep_rates[6])

    conv8 = subnet_a(conv7, [3,3,350,400], phase_train, bn_decay, func, keep_rates[7])
    
    pool4 = Max_pool(conv8, padding='VALID')

    fc_input = tf.reshape(pool4, [-1, 1600])

    fc1 = subnet_b(fc_input, 1600, 900, phase_train, bn_decay, func, keep_rates[8])

    fc2 = subnet_b(fc1, 900, 200, phase_train, bn_decay, func, keep_rates[9])

    logits = FC(fc2, 200, 3755, regular=True)

    return fc2, logits

