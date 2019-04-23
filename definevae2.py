# K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
# code for reproducing "MR image reconstruction using deep density priors"
# IEEE TMI, 2018
# tezcan@vision.ee.ethz.ch, CVL ETH Zürich



def definevae2(lat_dim=60, patchsize=28,batchsize=50):
 
     import tensorflow as tf
     config=tf.ConfigProto()
     config.gpu_options.allow_growth=True
     config.allow_soft_placement=True 
     import numpy as np
     import os
         
     # some internal parameter settings
     mode='MRIunproc'
     SEED=10    
     ndims=patchsize
     noisy=50
     batch_size = batchsize    
     nzsamp=1
     input_dim=ndims*ndims
     fcl_dim=500 
     num_inp_channels=1
     
     
     # standard deviation for the network weight initializer
     std_init=0.05               

    
     print("KCT-info: lat_dim value: "+str(lat_dim))
     print("KCT-info: mode is: " + mode)
          
     lat_dim_1 = max(1, np.floor(lat_dim/2))
     
     # necessary to tell TF which GPU to use
     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
     from tensorflow.python.client import device_lib
     print (device_lib.list_local_devices())
     print( os.environ['SGE_GPU'])
     
     
     #make a fully connected network
     #==============================================================================
     #==============================================================================
     
     tf.reset_default_graph()
     
     #define the activation function to use:
     def fact(x):
          return tf.nn.relu(x)
     
     
     #define the input place holder
     x_inp = tf.placeholder("float", shape=[None, input_dim])
     nsampl=50
     
     
     #define the weight initializer
     intl=tf.truncated_normal_initializer(stddev=std_init, seed=SEED)
     
     
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

     # unused, kept for compatibility
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
                      

     # f. build the output layer for precision matrix w/out activation function
     dec_out_cov = tf.nn.conv2d(dec_relu3, dec1_out_cov_weights, strides=[1, 1, 1, 1], padding='SAME')
     y_out_prec_log = tf.nn.bias_add(dec_out_cov, dec1_out_cov_biases)
     
     y_out_prec_ = tf.exp(y_out_prec_log)
     
     y_out_prec=tf.contrib.layers.flatten(y_out_prec_)
          
     
     # build the loss functions and the optimizer
     #==============================================================================
     #==============================================================================
     
     # KLD loss per sample in the batch
     KLD = -0.5 * tf.reduce_sum(1 + logVar - tf.pow(mu, 2) - tf.exp(logVar), reduction_indices=1)
     
     x_inp__ = tf.tile(x_inp, (nzsamp, 1))
     
     # L2 loss per sample in the batch
     l2_loss_1 = tf.reduce_sum(tf.multiply(tf.pow((y_out - x_inp__),2), y_out_prec),axis=1)
     l2_loss_2 = tf.reduce_sum(tf.log(y_out_prec), axis=1)
     l2_loss_ = l2_loss_1 - l2_loss_2

          
     
     # start session
     #==============================================================================
     #==============================================================================
     sess=tf.InteractiveSession(config=config)

     sess.run(tf.global_variables_initializer())
     print("KCT-info: Initialized parameters")
        
     saver = tf.train.Saver()
     
     
     # do post-training predictions
     #==============================================================================
     #==============================================================================
     print("KCT-info: restoring the l2 model, high resolution")
     saver.restore(sess, './trained_model/cvae_MSJhalf_'+mode+'_fcl'+str(fcl_dim)+'_lat'+str(lat_dim)+'_ns'+str(noisy)+'_ps'+str(patchsize))

     
     #Here I make a new variable called x_rec to replace the existing x_inp.
     #Then I recreate the graph with x_rec as the input so that I can later
     #take the derivative according to the input x_rec.
     #==============================================================================
     #==============================================================================
     nsampl=batchsize
     x_rec=tf.get_variable('x_rec',shape=[nsampl,ndims*ndims],initializer=tf.constant_initializer(value=0.0))
     
     
     #REWIRE THE GRAPH
     #==============================================================================
     #==============================================================================
     
     # rewire the graph input
     x_inp_ = tf.reshape(x_rec, [nsampl,ndims,ndims,1])
     
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
     z = mu + tf.multiply(std, epsilon) # z_std_multip*epsilon     #   # KCT!!!  
     
     # unused, kept for backward compatibility
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
                      
     # f. build the output layer for precision matrix w/out activation function
     dec_out_cov = tf.nn.conv2d(dec_relu3, dec1_out_cov_weights, strides=[1, 1, 1, 1], padding='SAME')
     y_out_prec_log = tf.nn.bias_add(dec_out_cov, dec1_out_cov_biases)
     
     y_out_prec_ = tf.exp(y_out_prec_log)
     
     y_out_prec=tf.contrib.layers.flatten(y_out_prec_)
                  
     #REWIRING THE GRAPH FINISHED
     #==============================================================================
     #==============================================================================
     
     
     # define the operations needed for the gradients and the pradients
     #==============================================================================
     #==============================================================================
     
     op_p_x_z = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_rec),2), y_out_prec),axis=1) \
                  + 0.5 * tf.reduce_sum(tf.log(y_out_prec), axis=1) -  0.5*ndims*ndims*tf.log(2*np.pi) ) 
     
     op_q_z_x = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z - mu),2), tf.reciprocal(std)),axis=1) \
                       - 0.5 * tf.reduce_sum(tf.log(std), axis=1) -  0.5*lat_dim*tf.log(2*np.pi))
     
     z_pl = tf.get_variable('z_pl',shape=[nsampl,lat_dim],initializer=tf.constant_initializer(value=0.0))
     
     op_q_zpl_x = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z_pl - mu),2), tf.reciprocal(std)),axis=1) \
                       - 0.5 * tf.reduce_sum(tf.log(std), axis=1) -  0.5*lat_dim*tf.log(2*np.pi))
     
     op_p_z = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z - tf.zeros_like(mu)),2), tf.reciprocal(tf.ones_like(std))),axis=1) \
                       - 0.5 * tf.reduce_sum(tf.log(tf.ones_like(std)), axis=1) -  0.5*lat_dim*tf.log(2*np.pi))
     
     
     funop=op_p_x_z + op_p_z - op_q_z_x
     
     grd = tf.gradients(op_p_x_z + op_p_z - op_q_z_x, x_rec) # 
     grd_p_x_z0 = tf.gradients(op_p_x_z, x_rec)[0]
     grd_p_z0 = tf.gradients(op_p_z, x_rec)[0]
     grd_q_z_x0 = tf.gradients(op_q_z_x, x_rec)[0]
     
     grd_q_zpl_x_az0 = tf.gradients(op_q_zpl_x, z_pl)[0]
     
     grd2 = tf.gradients(grd[0], x_rec)
     
     print("KCT-INFO: the gradients: ")
     print(grd_p_x_z0)
     print(grd_p_z0)
     print(grd_q_z_x0)
     
     grd0=grd[0]
     grd20=grd2[0]
                                                          
          
     return x_rec, x_inp, funop, grd0, sess, grd_p_x_z0, grd_p_z0, grd_q_z_x0, grd20, y_out, y_out_prec, op_q_z_x, mu, std, grd_q_zpl_x_az0, op_q_zpl_x, z_pl, z                               





