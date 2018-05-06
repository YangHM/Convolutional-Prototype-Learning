# Convolutional-Prototype-Learning
An implementation (TensorFlow) of CPL and GCPL appeared in CVPR2018 paper: "Robust Classification with Convolutional Prototype Learning";

The "mnist.data" is the processed MNIST dataset from "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz", I re-save them with cPickle in python for conveniently reading and loading, please contact me for the whole dataset; 

Type "python mnist_cpl.py" or "python mnist_gcpl.py" to run the program, you can change the settings with the command line parameters, for example, type "python mnist_gcpl.py --learning_rate=0.001" to change the initial learning rate ;

For the problems (or the datasets), please contact: hongming.yang@nlpr.ia.ac.cn
