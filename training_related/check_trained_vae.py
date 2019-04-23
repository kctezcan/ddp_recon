#
## do post-training predictions
##==============================================================================
##==============================================================================
#
#
#
#test_batch = DS.get_test_batch(batch_size)
#
#saver.restore(sess, '/home/ktezcan/modelrecon/trained_models/cvae_MSJ_'+mode+'_fcl'+str(fcl_dim)+'_lat'+str(lat_dim)+'_ns'+str(noisy)+'_ps'+str(ndims))
#
#     
##xh = y_out.eval(feed_dict={x_inp: test_batch}) 
##xch = 1./y_out_prec.eval(feed_dict={x_inp: test_batch}) 
#
#xh, xch = sess.run([y_out, 1/y_out_prec], feed_dict={x_inp: test_batch})
#
#nsamp=50
#
#yos=np.zeros((nsamp,input_dim))
#sos=np.zeros((nsamp,input_dim))
#yos_samp = np.zeros((nsamp,input_dim))
#for ix in range(nsamp):
#     print(ix)
#     zr = np.random.randn(1,lat_dim)
#     yo=y_out.eval(feed_dict={z: np.tile(zr,[50,1])})
#     try:
#          so=1./y_out_prec.eval(feed_dict={z: np.tile(zr,[50,1])})
#     except:
#          so = 1/kld_div          
#     yos[ix,:]=yo[0]
#     sos[ix,:]=np.sqrt(so[0])
#     yos_samp[ix,:] = yos[ix,:] + np.random.randn(input_dim)*sos[ix,:]
#
#
#print("generated means: ")
#print("=========================")
#plt.figure(figsize=(10,10))
#for ix in range(16):
#    plt.subplot(4,4, ix+1);
#    plt.imshow(np.reshape(yos[ix,:],(28,28)),cmap='gray');plt.xticks([]);plt.yticks([])
#
#
#print("generated covs: ")
#print("=========================")
#plt.figure(figsize=(10,10))
#for ix in range(16):
#    plt.subplot(4,4, ix+1)
#    plt.imshow(np.reshape(sos[ix,:],(28,28)),cmap='gray')
#
#
#print("means + [-1,+1]*covs: ")
#print("=========================")
#show_samp=20
#mults = np.linspace(-40,40,7)
#fig, ax = plt.subplots(show_samp,9, figsize=(20,show_samp*2))
#for ix in range(show_samp):
#    for ixc in range(7):
#         ax[ix][ixc].imshow(np.reshape(xh[ix,:],(28,28))+mults[ixc]*np.reshape(xch[ix,:],(28,28)),cmap='gray',vmin=-0.2,vmax=1.2)
#         ax[ix][7].imshow(np.reshape(test_batch[ix,:],(28,28)), cmap='gray',vmin=-0.2,vmax=1.2)
#         ax[ix][8].imshow(np.reshape(xch[ix,:],(28,28)), cmap='gray')
#
#
#
#
#
#
##########
##
###person=DS.get_test_batch(1)  
###person=denoise[80:108,80:108]
##
##
##from scipy import ndimage
##
##x_rec2=tf.get_variable('x_rec2',shape=[50,input_dim],initializer=tf.constant_initializer(value=0.0))
##
##person=DS.MRi.d_brains_test[10,180:208,180:208]
###person=denoise[180:208,180:208]
##
##nsampl=50
##imsize=28
##minperc=2
##maxperc=98
##nsigma=15
##   
##op_p_x_z = - 0.5 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_rec2),2), y_out_prec),axis=1) \
##             + 0.5 * tf.reduce_sum(tf.log(y_out_prec), axis=1) #-  0.5*48*48*np.log(2*np.pi)
##       
##op_p_x_z_0 = - 0.5 * tf.reduce_sum(tf.pow((y_out - x_rec2),2),axis=1)      
##op_p_x_z_1 = - 0.5 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_rec2),2), y_out_prec),axis=1)
##op_p_x_z_2 = + 0.5 * tf.reduce_sum(tf.log(y_out_prec), axis=1)
##
##op_q_z_x = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z - mu),2), tf.reciprocal(std)),axis=1) \
##                  - 0.5 * tf.reduce_sum(tf.log(std), axis=1) -  0.5*lat_dim*tf.log(2*np.pi))
##
##op_p_z = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z - tf.zeros_like(mu)),2), tf.reciprocal(tf.ones_like(std))),axis=1) \
##                  - 0.5 * tf.reduce_sum(tf.log(tf.ones_like(std)), axis=1) -  0.5*lat_dim*tf.log(2*np.pi))
##
##
##np.set_printoptions(threshold=1000)
##
##sigmas=np.linspace(0.001,30,nsigma)
##
##tbbs=np.zeros((nsigma,imsize,imsize))
##
##goodbad=np.zeros(25)
##persons=np.zeros((25,imsize,imsize))
##
##plt.figure(figsize=(20,10))
##
##for ixm in range(10,11):
##     person=DS.get_test_batch(1) 
##     
##     aas=np.zeros((nsigma,1))#,dtype='float64')
##     Ks=np.zeros((nsigma,1))#,dtype='float64')
##     aams = np.zeros((nsigma,1))#,dtype='float64')
##     bbs=np.zeros((nsampl,nsigma))#,dtype='float64')
##
##     for ix in range(nsigma):
##         print(ix)
##         tb=np.reshape(person,(imsize,imsize)) 
##         tbb=ndimage.gaussian_filter(tb,sigma=sigmas[ix])
##         #tbb=tb + np.random.normal(loc=0, scale=sigmas[ix], size=tb.shape)
##         tbb=(tbb - np.percentile(tbb, minperc))/(np.percentile(tbb, maxperc) - np.percentile(tbb, minperc))
##         tbbs[ix,:,:]=tbb.copy()
##         tbb=np.reshape(tbb,(1,imsize*imsize))
##         x_rec2.load(    value = np.tile(tbb,(nsampl,1))    )
##         #get p(x|z_n), q(z_n|x) and p(z_n)
##         p_x_z, q_z_x, p_z  = sess.run([op_p_x_z, op_q_z_x, op_p_z], feed_dict={x_inp: np.tile(tbb,(nsampl,1))})
##     
##         aa = (p_x_z - q_z_x + p_z).astype('float128') #).astype('float128') 
##         
##         K=np.max(aa)
##         
##         aad=(aa-K).astype('float128')
##         
##         aae=np.exp(aad)
##         
##         aaes=np.sum(aae)
##         if ix==31:
##              print("aa")
##              print(aa)
##              print("aae")
##              print(aae)
##              print("aaes")
##              print(aaes)
##         
##         
##         aaesl=np.log(aaes)
##         
##         aaeslKS=aaesl + K - np.log(nsampl)
##     
##         aas[ix,0]=aaeslKS
##            
##           
##     if aas[0]>aas[-1]:
##          goodbad[ixm]=1
##     else:
##          goodbad[ixm]=0
##     persons[ixm,:,:]=np.reshape(person,(imsize,imsize))
##     
##     
##     plt.plot(sigmas, aas,'.-')
##
##
##
##
##
##
###from tensorflow.examples.tutorials.mnist import input_data
###from scipy import ndimage
###mnist2 = input_data.read_data_sets('MNIST')
###
###aas=np.zeros((50,1))
###sigmas=np.linspace(0.001,1,50)
###for ix in range(50):
###    tb=np.reshape(mnist2.test.images[0,:],(28,28)) 
###    tbb=ndimage.gaussian_filter(tb,sigma=sigmas[ix])
###    tbb=np.reshape(tbb,(1,784))
###    x_rec.load(    value = np.tile(tbb,(5000,1))    )
###    aa = (- 1/2 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_rec),2), y_out_prec),axis=1) \
###             + 1/2 * tf.reduce_sum(tf.log(y_out_prec), axis=1) -  1/2*784*tf.log(2*np.pi)).eval(feed_dict={x_inp: np.tile(tbb,(5000,1))})
###    aas[ix,0]=aa.mean()
###
###plt.plot(sigmas, aas, '.-'), plt.title("log(p(x)) vs. sigma of blurrung gaussian"), plt.xlabel("gaussian std"),plt.ylabel("log likelihood")
##     
##     
##     
###
#### do post-training predictions
####==============================================================================
####==============================================================================
###
###test_batch = DS.get_test_batch(batch_size)
###
###if user == 'Kerem':
###     saver.restore(sess, '/home/ktezcan/Code/spyder_files/tests_code_results/models_vae_mri_ms_iter20k/vae_MG_fcl'+str(fcl_dim)+'_lat'+str(lat_dim)+'_ns'+str(noisy)+'_klddiv'+str(int(kld_div)))
###elif user == 'Ender':     
###     saver.restore(sess,'/scratch/kender/Projects/VAE/spyder_files')
###else:
###     print("User unknown!")
###     assert(1==0)
###     
###xh = y_out.eval(feed_dict={x_inp: test_batch}) 
###xch = 1./y_out_prec.eval(feed_dict={x_inp: test_batch}) 
###
###
###nsamp=5000
###
###yos=np.zeros((nsamp,input_dim))
###sos=np.zeros((nsamp,input_dim))
###yos_samp = np.zeros((nsamp,input_dim))
###for ix in range(nsamp):
###     zr = np.random.randn(1,lat_dim)
###     yo=y_out.eval(feed_dict={z: zr})
###     try:
###          so=1./y_out_prec.eval(feed_dict={z: zr})
###     except:
###          so = 1/kld_div          
###     yos[ix,:]=yo
###     sos[ix,:]=np.sqrt(so)
###     yos_samp[ix,:] = yos[ix,:] + np.random.randn(input_dim)*sos[ix,:]
###
###
###print("generated means: ")
###print("=========================")
###plt.figure(figsize=(10,10))
###for ix in range(16):
###    plt.subplot(4,4, ix+1)
###    plt.imshow(np.reshape(yos[ix,:],(28,28)),cmap='gray')
###
###
###print("generated covs: ")
###print("=========================")
###plt.figure(figsize=(10,10))
###for ix in range(16):
###    plt.subplot(4,4, ix+1)
###    plt.imshow(np.reshape(sos[ix,:],(28,28)),cmap='gray')
###
###
###print("means + [-1,+1]*covs: ")
###print("=========================")
###show_samp=20
###mults = np.linspace(-1,1,7)
###fig, ax = plt.subplots(show_samp,9, figsize=(20,show_samp*2))
###for ix in range(show_samp):
###    for ixc in range(7):
###         ax[ix][ixc].imshow(np.reshape(xh[ix,:],(28,28))+mults[ixc]*np.reshape(xch[ix,:],(28,28)),cmap='gray')
###         ax[ix][7].imshow(np.reshape(test_batch[ix,:],(28,28)), cmap='gray')
###         ax[ix][8].imshow(np.reshape(xch[ix,:],(28,28)), cmap='gray',vmin=-1, vmax=1)
###
###
###
###from tensorflow.examples.tutorials.mnist import input_data
###from scipy import ndimage
###mnist2 = input_data.read_data_sets('MNIST')
###
###aas=np.zeros((50,1))
###sigmas=np.linspace(0.001,1,50)
###for ix in range(50):
###    tb=np.reshape(mnist2.test.images[0,:],(28,28)) 
###    tbb=ndimage.gaussian_filter(tb,sigma=sigmas[ix])
###    tbb=np.reshape(tbb,(1,784))
###    x_rec.load(    value = np.tile(tbb,(5000,1))    )
###    aa = (- 1/2 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_rec),2), y_out_prec),axis=1) \
###             + 1/2 * tf.reduce_sum(tf.log(y_out_prec), axis=1) -  1/2*784*tf.log(2*np.pi)).eval(feed_dict={x_inp: np.tile(tbb,(5000,1))})
###    aas[ix,0]=aa.mean()
###
###plt.plot(sigmas, aas, '.-'), plt.title("log(p(x)) vs. sigma of blurrung gaussian"), plt.xlabel("gaussian std"),plt.ylabel("log likelihood")
##     
##     
##     