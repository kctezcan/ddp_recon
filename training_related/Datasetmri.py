# K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
# code for reproducing "MR image reconstruction using deep density priors"
# IEEE TMI, 2018
# tezcan@vision.ee.ethz.ch, CVL ETH Zürich


from __future__ import division
from __future__ import print_function

#import os.path

import numpy as np
import scipy.misc
import glob, os
import h5py


class Dataset:

     def __init__(self, train_size, test_size, ndims, noisy, seed, mode, downscale=False, useslice=[130, 135, 140, 145, 150]):

          np.random.seed(seed=seed)

          self.train_size = train_size
          self.test_size = test_size
          self.mode = mode
          self.noise_std = noisy
          self.patch_size = ndims
          
          self.downscale = downscale
          
          self.useslice=useslice
          
          self.percentage = 0
          

   
               
          from MR_image_data import MR_image_data
          
          dirname='/scratch_net/bmicdl02/Data/hcpdata_ribbon_brain/'
          imgSize=[240, 300]
          useSlice=self.useslice
          showStuff=False
          self.MRi = MR_image_data(dirname, imgSize, useSlice, 99, showStuff, withSkullStrip=False, withBFCorrect = False , downscale=self.downscale)
          self.MRi.read_images(read_new=True)
          self.MRi.generate_US_train_test_images(percTrain=90, R=1, fullySampledCenter=11, generateNew=True)
          self.train_set=[]
          self.test_set=[]


     def get_train_batch(self, batch_size):
                    
          if   self.mode == 'MNIST':
               aa = self.mnist.train.next_batch(batch_size)[0]
               return aa + np.random.normal(loc=0, scale=1/self.noise_std, size=aa.shape)
          
          elif self.mode == 'CelebA':     
               
               #make sure you get one!
               success=False
               
               while not success: 
                    try:
                         rr=np.random.randint(low=0, high=int(0.9*len(self.file_names)), size=batch_size)
                         rr.sort()
                         
                         h5f = h5py.File('/scratch_net/bmicdl02/CelebA/'+'celeba_' + str(self.patch_size) + '.h5','r')
                         batch = h5f['ims'][rr,:,:]
                         
                         success=True
                         
                    except:
                         pass
                    
               if self.noise_std:
                    batch = batch + np.random.normal(loc=0, scale=1/self.noise_std, size=batch.shape)
               
               return np.reshape(batch,(batch_size,-1))
          
          elif self.mode == 'CelebAP':
               
               #make sure you get one!
               success=False
               
               batch=np.zeros((batch_size,self.patch_size,self.patch_size))
               
               while not success: 
                    try:
                         rr=np.random.randint(low=0, high=int(0.9*len(self.file_names)), size=batch_size)
                         rrx=np.random.randint(low=0, high=int(116-self.patch_size), size=batch_size)
                         rry=np.random.randint(low=0, high=int(80-self.patch_size), size=batch_size)
                         rr.sort()
                         
                         h5f = h5py.File('/scratch_net/bmicdl02/CelebA/'+'celeba_' + '116x80'+ '.h5','r')
                         for ix in range(batch_size):
                              
                              tmp = h5f['ims'][rr[ix],rrx[ix]:rrx[ix]+self.patch_size,rry[ix]:rry[ix]+self.patch_size]
                              batch[ix,:,:]=tmp
                         success=True
                         
                    except:
                         pass
                    
                    return np.reshape(batch, (batch_size,-1))
          
          elif self.mode=='MRI' or self.mode=='MRIunproc' or self.mode=='MRIonlybfc':
#               test_images_us, test_images_fs, test_labels = self.MRi.get_batch_patch_usfs_segm(batch_size, self.patch_size, 20 ,test=True)
               train_images_us, train_images_fs, train_labels = self.MRi.get_batch_patch_usfs_segm(batch_size, self.patch_size, self.percentage ,test=False)
               
               if self.noise_std:
                    train_images_fs = train_images_fs + np.random.normal(loc=0, scale=1/self.noise_std, size=train_images_fs.shape)
               
               return np.reshape(train_images_fs,(batch_size,-1))
          
          else:
               rr=np.random.randint(low=0, high=self.train_size, size=batch_size)
               return self.train_set[rr,:]


     def get_test_batch(self, batch_size):
          if self.mode == 'MNIST':
               aa=self.mnist.test.next_batch(batch_size)[0]
               return aa + np.random.normal(loc=0, scale=1/self.noise_std, size=aa.shape)
          
          elif self.mode== 'CelebA':
               
                #make sure you get one!
                success=False
               
                while not success:
                    try:
                         rr=np.random.randint(low=int(0.9*len(self.file_names)), high=len(self.file_names), size=batch_size)
                         rr.sort()
                         
                         h5f = h5py.File('/scratch_net/bmicdl02/CelebA/'+'celeba_' + str(self.patch_size) + '.h5','r')
                         batch = h5f['ims'][rr,:,:]
                         
                         success = True
                    except:
                         pass
               
                if self.noise_std:
                    batch = batch + np.random.normal(loc=0, scale=1/self.noise_std, size=batch.shape)
                    
                return np.reshape(batch,(batch_size,-1)) 
          
          elif self.mode == 'CelebAP':
               
               #make sure you get one!
               success=False
               
               batch=np.zeros((batch_size,self.patch_size,self.patch_size))
               
               while not success: 
                    try:
                         rr=np.random.randint(low=int(0.9*len(self.file_names)), high=len(self.file_names), size=batch_size)
                         rrx=np.random.randint(low=0, high=int(116-self.patch_size), size=batch_size)
                         rry=np.random.randint(low=0, high=int(80-self.patch_size), size=batch_size)
                         rr.sort()
                         
                         h5f = h5py.File('/scratch_net/bmicdl02/CelebA/'+'celeba_' + '116x80'+ '.h5','r')
                         for ix in range(batch_size):
                              
                              tmp = h5f['ims'][rr[ix],rrx[ix]:rrx[ix]+self.patch_size,rry[ix]:rry[ix]+self.patch_size]
                              batch[ix,:,:]=tmp
                         success=True
                         
                    except:
                         pass
                    
                    return np.reshape(batch,(batch_size,-1)) 
          
          elif self.mode=='MRI' or self.mode=='MRIunproc' or self.mode=='MRIonlybfc':
               test_images_us, test_images_fs, test_labels = self.MRi.get_batch_patch_usfs_segm(batch_size, self.patch_size, self.percentage ,test=True)
               
               if self.noise_std:
                    test_images_fs = test_images_fs + np.random.normal(loc=0, scale=1/self.noise_std, size=test_images_fs.shape)
               
               return np.reshape(test_images_fs,(batch_size,-1))
          
          else:          
               rr=np.random.randint(low=0, high=self.test_size, size=batch_size)
               return self.test_set[rr,:]
          
     
#-endfunc
             
#     def get_test_batch_th(self, batch_size):
#          if self.mode == 'CelebAP':
#               batch_thread = threading.Thread(target=self.get_test_batch, args=batch_size)
#               batch_thread.start()
#          else:
#               print("mode unknown or not allowed!")
#               assert(1==0)
#               
#     def get_train_batch_th(self, batch_size):
#          if self.mode == 'CelebAP':
#               batch_thread = threading.Thread(target=self.get_train_batch, args=batch_size)
#               batch_thread.start()
#          else:
#               print("mode unknown or not allowed!")
#               assert(1==0)
#-endclass
     
#     def iterate_train_batches()
     


# produce l1-sparse 2D samples on the axes of the 2D space
def get_batch_l1(batch_size,ndim=2,noisy=False):

     # choose which axis to be non-zero
     r=np.floor( ndim*np.random.rand(batch_size))

     # produce a zeros array for n-dimensional vectors
     b=np.zeros((batch_size,ndim))

     # draw samples from a distribution
     #w=-1+ 2*np.random.rand(batch_size)
     w= np.random.randn(batch_size)

     # put the samples into one of the axes
     for inx in range(0,batch_size):
          b[inx,r[inx].astype(int)]=w[inx]

     if noisy:
          b=b+np.random.normal(loc=0, scale=1/noisy, size=b.shape)

     return b

# produce non-sparse 2D samples on the axes of the 2D space
def get_batch_l2(batch_size,ndim=2,noisy=False):

     #produce full samples
     b=1*np.random.randn(batch_size, ndim)
     #b = -1 + 2*np.random.rand(batch_size,2)


     if noisy:
          b=b+np.random.normal(loc=0, scale=1/noisy, size=b.shape)

     return b

# produce non-sparse 2D samples on the axes of the 2D space
def get_batch_Lshape(batch_size,ndim=2,noisy=False):

     ndim=ndim-1

     #produce full samples
     #b=1*np.random.randn(batch_size, ndim)
     b=1*np.random.laplace(loc=0.0, scale=1.0, size=(batch_size, ndim))
     #b = -1 + 2*np.random.rand(batch_size,2)


#     a=np.array([3,5])
#     b= b * a[np.newaxis,:]
     a = np.zeros([b.shape[0],2])
     a[b[:,0]<0,0] = b[b[:,0]<0,0]
     a[b[:,0]>=0,1] = b[b[:,0]>=0,0]

     if noisy:
          b=b+np.random.normal(loc=0, scale=1/noisy, size=b.shape)

     return a

def get_circular (batch_size,ndim=2,noisy=False):
    vec = np.random.randn(batch_size,ndim)
    vec /= np.linalg.norm(vec, axis=1)[:,np.newaxis]
    vec=vec+np.random.normal(loc=0, scale=1/noisy, size=vec.shape)
    return vec

def get_batch_edge(batch_size, ndim=15, noisy=False):

     bbs=np.zeros((batch_size,ndim))

     for ix in range(batch_size):
          ep=np.random.randint(low=0, high=ndim, size=1)[0]
          #ep = 7
          ip1=np.random.randn(1)
          ip2=np.random.randn(1)
          #ip1=1
          #ip2=0

          bb=np.ones(shape=(1,ndim))*ip1
          bb[0,ep:]=ip2
          bbs[ix,:]=bb[0,:]
     #-endfor

     if noisy:
          bbs = bbs + np.random.normal(loc=0, scale=1/noisy, size=bbs.shape)
     #endif

     return bbs
#-endfunc

def get_batch_Laplacian_edge(batch_size, ndim=15, noisy=False):
     scale=0.3
     a = np.random.laplace(scale=scale, size=[batch_size, ndim])
     A = np.cumsum(a**3, axis=1)

     if noisy:
          A = A + np.random.normal(loc=0, scale=1/noisy, size=A.shape)

     return A
#-endfunc

def get_batch_cloud_train(batch_size, ndim=2, noisy=False):
     
     crn=16

     all=np.zeros((batch_size,2))
     all[0:int(batch_size/4),:]=np.random.multivariate_normal([0,0], [[0.1,0],[0,0.1]], size=int(batch_size/4) ) 
     all[int(batch_size/4):int(2*batch_size/4),:]=np.random.multivariate_normal([0,crn], [[0.1,0],[0,0.1]], size=int(batch_size/4) ) 
     all[int(2*batch_size/4):int(3*batch_size/4),:]=np.random.multivariate_normal([crn,0], [[0.1,0],[0,0.1]], size=int(batch_size/4) ) 
     all[int(3*batch_size/4):,:]=np.random.multivariate_normal([crn,crn], [[0.1,0],[0,0.1]], size=int(batch_size/4) ) 
     
     all=all-[crn/2,crn/2]
     
     if noisy:
          all = all + np.random.normal(loc=0, scale=1/noisy, size=all.shape)

     return all
#-endfunc


def get_batch_chirp_train(batch_size, ndim=2, noisy=False):

     all=np.random.laplace(loc=0,scale=1,size=(batch_size,2))**2
     
     
     if noisy:
          all = all + np.random.normal(loc=0, scale=1/noisy, size=all.shape)

     return all
#-endfunc


def get_batch_Gaussian_edge(batch_size, ndim=15, noisy=False):
     scale=0.3
     a = np.random.normal(scale=scale, size=[batch_size, ndim])
     A = np.cumsum(a, axis=1)

     if noisy:
          A = A + np.random.normal(loc=0, scale=1/noisy, size=A.shape)

     return A
 # endfunc

def get_batch_spike_train(batch_size, ndim=150, noisy=False):
     ns=np.floor(ndim/15).astype(int)
     assert( ns > 0)

     A=np.zeros((batch_size, ndim))

     for ix in range(batch_size):
          a = np.random.randint(low=0, high=ndim, size=(ns))
          A[ix,a]=np.random.randn()

     if noisy:
          A = A + np.random.normal(loc=0, scale=1/noisy, size=A.shape)

     return A
#-endfunc

#copy pasted from carpedm20's dcgan implementation
def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
     
     offset=15
     
     if crop_w is None:
          crop_w = crop_h
     h, w = x.shape[:2]
     j = int(round((h - crop_h)/2.))
     i = int(round((w - crop_w)/2.))
     
     if (resize_h==crop_h) and (resize_w==crop_w) :
          return x[j+offset:j+offset+crop_h, i:i+crop_w]
     else:
          return scipy.misc.imresize(x[j+offset:j+offset+crop_h, i:i+crop_w], [resize_h, resize_w], interp='lanczos')
#-endfunc

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
     if crop:
          cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
     else:
          cropped_image = scipy.misc.imresize(image, [resize_height, resize_width], interp='lanczos')
          
     return np.array(cropped_image)/255.0
#-endfunc

def get_image(image_path, input_height, input_width, resize_height=40, resize_width=40, crop=True, grayscale=True):
     image = imread(image_path, grayscale)
     return transform(image, input_height, input_width, resize_height, resize_width, crop)
#-endfunc

def imread(path, grayscale = True):
     if (grayscale):
          return scipy.misc.imread(path, flatten = True).astype(np.float)
     
     else:
          return scipy.misc.imread(path).astype(np.float)
#-endfunc



#
#
#def im2patches(fim, patschsize, batchsize):
#    #make patches from full image
#
#    B = view_as_blocks( fim[0:np.floor_divide(fim.shape[0],patchsize)*patchsize , 0:np.floor_divide(fim.shape[1],patchsize)*patchsize] , block_shape=(patschsize, patschsize))
#
#    Bb=np.reshape(B,(-1, patschsize, patschsize ) )
#    zs=np.zeros((batchsize-Bb.shape[0], patschsize, patschsize))
#    Bb=np.concatenate((Bb,zs), axis=0 )
#
#    Bb=np.reshape(Bb,(batchsize,patchsize,patchsize,1))
#
#    return Bb
##-endfunc
#
#
#def patches2im(patches, patchsize, orimsize):
#    
#    patches=patches[0:int(np.prod(orimsize)/patchsize**2),:,:]
#    patches = np.squeeze(patches)
#    gbbr=np.reshape(patches, (  int(orimsize[0]/patchsize), int(orimsize[1]/patchsize), patchsize, patchsize  ) )
#    
#    blk=np.zeros( (240,288))
#    for rr in range(int(orimsize[0]/patchsize)):
#        for cc in range(int(orimsize[1]/patchsize)):
#            blk[ (rr*patchsize):((rr+1)*patchsize), (cc*patchsize):((cc+1)*patchsize)   ] =  gbbr[rr,cc,:,:]
#    
#    return blk
##-endfunc
#                           
#
#
#














