# K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
# code for reproducing "MR image reconstruction using deep density priors"
# IEEE TMI, 2018
# tezcan@vision.ee.ethz.ch, CVL ETH Zürich


# <<<<<<<<<<<<<<<<<<< README <<<<<<<<<<<<<<<<<<<
# to make the code run, you need a dataset. Here I have marked the lines where
# we load and use the dataset with "# Modify here".
#
# Please replace those lines with your dataset
# module, or modify the of the code to make it work with your dataset.
#
# Notice the method works with patches, not with full size images. Hence the 
# dataset module has to provide patches from your training dataset.
# <<<<<<<<<<<<<<<<<<< README <<<<<<<<<<<<<<<<<<<


from __future__ import division
from __future__ import print_function
#import os.path

import numpy as np
import time as tm
import tensorflow as tf
import os
from YourDatasetModuleHere import YourDatasetModuleHere # Modify here


SEED=1001
seed=1
np.random.seed(seed=seed)


# parameters
#==============================================================================
#==============================================================================
mode='MRIunproc'

ndims=28 # patch size
noisy=50 # noise std added on the images during training
batch_size = 50 
nzsamp=1 # for using more than one z, not tested


std_init=0.05   # for the weight inititalizers             

input_dim=ndims*ndims
fcl_dim=500 # not functional, kept for backward compatibility

lat_dim=60 # dimensions of the latent space
     
lat_dim_1 = max(1, np.floor(lat_dim/2)) # unfunctional, kept for backward compatibility
lat_dim_2 = lat_dim - lat_dim_1

num_inp_channels=1

# necessary for telling TF which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())
print( os.environ['SGE_GPU'])

#make a dataset to use later - load your dataset module here
#==============================================================================
#==============================================================================
DS = YourDatasetModuleHere() # Modify here


#make a network
#==============================================================================
#==============================================================================
tf.reset_default_graph()
sess=tf.InteractiveSession()


#define the activation function to use:
def fact(x):
     return tf.nn.relu(x)


#define the input place holder
x_inp = tf.placeholder("float", shape=[None, input_dim])
l2_loss = tf.constant(0.0)

#define the network layer parameters
intl=tf.truncated_normal_initializer(stddev=std_init, seed=SEED)
intl_cov=tf.truncated_normal_initializer(stddev=std_init, seed=SEED)

with tf.variable_scope("VAE") as scope:
     
    enc_conv1_weights = tf.get_variable("enc_conv1_weights", [3, 3, num_inp_channels, 32], initializer=intl)
    enc_conv1_biases = tf.get_variable("enc_conv1_biases", shape=[32], initializer=tf.constant_initializer(value=0))
     
    enc_conv2_weights = tf.get_variable("enc_conv2_weights", [3, 3, 32, 64], initializer=intl)
    enc_conv2_biases = tf.get_variable("enc_conv2_biases", shape=[64], initializer=tf.constant_initializer(value=0))
     
    enc_conv3_weights = tf.get_variable("enc_conv3_weights", [3, 3, 64, 64], initializer=intl)
    enc_conv3_biases = tf.get_variable("enc_conv3_biases", shape=[64], initializer=tf.constant_initializer(value=0))
         
    mu_weights = tf.get_variable(name="mu_weights", shape=[int(input_dim*64), lat_dim], initializer=intl)
    mu_biases = tf.get_variable("mu_biases", shape=[lat_dim], initializer=tf.constant_initializer(value=0))
    
    logVar_weights = tf.get_variable(name="logVar_weights", shape=[int(input_dim*64), lat_dim], initializer=intl)
    logVar_biases = tf.get_variable("logVar_biases", shape=[lat_dim], initializer=tf.constant_initializer(value=0))
        
    dec_fc1_weights = tf.get_variable(name="dec_fc1_weights", shape=[int(lat_dim), int(input_dim*48)], initializer=intl)
    dec_fc1_biases = tf.get_variable("dec_fc1_biases", shape=[int(input_dim*48)], initializer=tf.constant_initializer(value=0))
    
    dec_conv1_weights = tf.get_variable("dec_conv1_weights", [3, 3, 48, 48], initializer=intl)
    dec_conv1_biases = tf.get_variable("dec_conv1_biases", shape=[48], initializer=tf.constant_initializer(value=0))
     
    dec_conv2_weights = tf.get_variable("decc_conv2_weights", [3, 3, 48, 90], initializer=intl)
    dec_conv2_biases = tf.get_variable("dec_conv2_biases", shape=[90], initializer=tf.constant_initializer(value=0))
     
    dec_conv3_weights = tf.get_variable("dec_conv3_weights", [3, 3, 90, 90], initializer=intl)
    dec_conv3_biases = tf.get_variable("dec_conv3_biases", shape=[90], initializer=tf.constant_initializer(value=0))
    
    dec_out_weights = tf.get_variable("dec_out_weights", [3, 3, 90, 1], initializer=intl)
    dec_out_biases = tf.get_variable("dec_out_biases", shape=[1], initializer=tf.constant_initializer(value=0))
    
    dec1_out_cov_weights = tf.get_variable("dec1_out_cov_weights", [3, 3, 90, 1], initializer=intl)
    dec1_out_cov_biases = tf.get_variable("dec1_out_cov_biases", shape=[1], initializer=tf.constant_initializer(value=0))

# The network
# a. build the encoder layers

x_inp_ = tf.reshape(x_inp, [batch_size,ndims,ndims,1])

enc_conv1 = tf.nn.conv2d(x_inp_, enc_conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
enc_relu1 = fact(tf.nn.bias_add(enc_conv1, enc_conv1_biases))

enc_conv2 = tf.nn.conv2d(enc_relu1, enc_conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
enc_relu2 = fact(tf.nn.bias_add(enc_conv2, enc_conv2_biases))

enc_conv3 = tf.nn.conv2d(enc_relu2, enc_conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
enc_relu3 = fact(tf.nn.bias_add(enc_conv3, enc_conv3_biases))
      
flat_relu3 = tf.contrib.layers.flatten(enc_relu3)

# b. get the values for drawing z
mu = tf.matmul(flat_relu3, mu_weights) + mu_biases
mu = tf.tile(mu, (nzsamp, 1)) # replicate for number of z's you want to draw
logVar = tf.matmul(flat_relu3, logVar_weights) + logVar_biases
logVar = tf.tile(logVar,  (nzsamp, 1))# replicate for number of z's you want to draw
std = tf.exp(0.5 * logVar)

# c. draw an epsilon and get z
epsilon = tf.random_normal(tf.shape(logVar), name='epsilon')
z = mu + tf.multiply(std, epsilon)

# usused, kept for backward compatibility
indices1=tf.range(start=0, limit=lat_dim_1, delta=1, dtype='int32')
indices2=tf.range(start=lat_dim_1, limit=lat_dim, delta=1, dtype='int32')
z1 = tf.transpose(tf.gather(tf.transpose(z),indices1))
z2 = tf.transpose(tf.gather(tf.transpose(z),indices2))

# d. build the decoder layers from z1 for mu(z)
dec_L1 = fact(tf.matmul(z, dec_fc1_weights) + dec_fc1_biases)     

dec_L1_reshaped = tf.reshape(dec_L1 ,[batch_size,int(ndims),int(ndims),48])

dec_conv1 = tf.nn.conv2d(dec_L1_reshaped, dec_conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
dec_relu1 = fact(tf.nn.bias_add(dec_conv1, dec_conv1_biases))

dec_conv2 = tf.nn.conv2d(dec_relu1, dec_conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
dec_relu2 = fact(tf.nn.bias_add(dec_conv2, dec_conv2_biases))

dec_conv3 = tf.nn.conv2d(dec_relu2, dec_conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
dec_relu3 = fact(tf.nn.bias_add(dec_conv3, dec_conv3_biases))

# e. build the output layer w/out activation function
dec_out = tf.nn.conv2d(dec_relu3, dec_out_weights, strides=[1, 1, 1, 1], padding='SAME')
y_out_ = tf.nn.bias_add(dec_out, dec_out_biases)

y_out = tf.contrib.layers.flatten(y_out_)
                   
# f. build the precision output layer w/out activation function
dec_out_cov = tf.nn.conv2d(dec_relu3, dec1_out_cov_weights, strides=[1, 1, 1, 1], padding='SAME')
y_out_prec_log = tf.nn.bias_add(dec_out_cov, dec1_out_cov_biases)

y_out_prec_ = tf.exp(y_out_prec_log)

y_out_prec=tf.contrib.layers.flatten(y_out_prec_)
     


# build the loss functions and the optimizer
#==============================================================================
#==============================================================================

# KLD loss per sample in the batch
KLD = -0.5 * tf.reduce_sum(1 + logVar - tf.pow(mu, 2) - tf.exp(logVar), reduction_indices=1)

x_inp_ = tf.tile(x_inp, (nzsamp, 1))

# L2 loss per sample in the batch
l2_loss_1 = tf.reduce_sum(tf.multiply(tf.pow((y_out - x_inp_),2), y_out_prec),axis=1)
l2_loss_2 = tf.reduce_sum(tf.log(y_out_prec), axis=1) #tf.reduce_sum(tf.log(y_out_cov),axis=1)
l2_loss_ = l2_loss_1 - l2_loss_2

# take the total mean loss of this batch
loss_tot = tf.reduce_mean(KLD + 0.5*l2_loss_)


train_step = tf.train.AdamOptimizer(5e-4).minimize(loss_tot)

# start session
#==============================================================================
#==============================================================================

sess.run(tf.global_variables_initializer())
print("Initialized parameters")

saver = tf.train.Saver()

ts=tm.time()

# do the training
#==============================================================================
#==============================================================================

test_batch = DS.get_test_batch(batch_size) # Modify here

with tf.device('/gpu:0'):
     
     #train for N steps
     for step in range(0, 250000):

         batch = DS.get_train_batch(batch_size) # Modify here
         
              
         # run the training step     
         sess.run([train_step], feed_dict={x_inp: batch})
         
    
         #print some results...
         if step % 500 == 0:
             loss_l2_1 = l2_loss_1.eval(feed_dict={x_inp: test_batch})
             loss_l2_2 = l2_loss_2.eval(feed_dict={x_inp: test_batch})
             loss_l2_ = l2_loss_.eval(feed_dict={x_inp: test_batch})
             loss_kld = KLD.eval(feed_dict={x_inp: test_batch})
             std_val = std.eval(feed_dict={x_inp: test_batch})
             mu_val = mu.eval(feed_dict={x_inp: test_batch})
             loss_tot_ = loss_tot.eval(feed_dict={x_inp: test_batch})
              
             
             xh = y_out.eval(feed_dict={x_inp: test_batch}) 
             test_loss_l2 = np. mean( np.sum(np.power((xh[0:test_batch.shape[0],:] - test_batch),2), axis=1) )
             

             print("Step {0} | L2 Loss: {1:.3f} | KLD Loss: {2:.3f} | L2 Loss_1: {3:.3f} | L2 Loss_2: {4:.3f} | loss_tot: {5:.3f} | L2 Loss test: {6:.3f}"\
                   .format(step, np.mean(loss_l2_1-loss_l2_2), np.mean(loss_kld), np.mean(loss_l2_1), np.mean(loss_l2_2), np.mean(loss_tot_), np.mean(test_loss_l2)))

# save the model into the current folder
saver.save(sess, './cvae_MSJhalf_'+mode+'_fcl'+str(fcl_dim)+'_lat'+str(lat_dim)+'_ns'+str(noisy)+'_ps'+str(ndims))

print("elapsed time: {0}".format(tm.time()-ts))


     