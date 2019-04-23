# K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
# code for reproducing "MR image reconstruction using deep density priors"
# IEEE TMI, 2018
# tezcan@vision.ee.ethz.ch, CVL ETH Zürich

import numpy as np
import os
import pickle
import vaerecon5

from US_pattern import US_pattern


# to tell tensorflow which GPU to use
#-----------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())
print( os.environ['SGE_GPU'])


#define the necessary encoding operations and their transposes
#-----------------------------------------------------
def FT (x):
     # coil expansion followed by Fourier transform 
     #inp: [nx, ny]
     #out: [nx, ny, ns]
     return np.fft.fftshift(    np.fft.fft2( sensmaps*np.tile(x[:,:,np.newaxis],[1,1,sensmaps.shape[2]]) , axes=(0,1)  ),   axes=(0,1)    )

def tFT (x):
     # inverse Fourier transform and coil combination
     #inp: [nx, ny, ns]
     #out: [nx, ny]
     
     temp = np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )
     return np.sum( temp*np.conjugate(sensmaps) , axis=2)  / np.sum(sensmaps*np.conjugate(sensmaps),axis=2)


def UFT(x, uspat):
     # Encoding: undersampling +  FT
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny, ns]
     
     return np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])*FT(x)

def tUFT(x, uspat):
     # transposed Encoding: inverse FT + undersampling
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny]
     
     tmp1 = np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])
     
     return  tFT( tmp1*x )

# RMSE calculation
#-----------------------------------------------------
def calc_rmse(rec,imorig):
     return 100*np.sqrt(np.sum(np.square(np.abs(rec) - np.abs(imorig))) / np.sum(np.square(np.abs(imorig))) )


#which VAE model to use
#-----------------------------------------------------
ndims=28 # patch size
lat_dim=60 # latent space size

# the US factor
#-----------------------------------------------------
R=2

#load the test image and make uniform coil maps
#-----------------------------------------------------
orim = np.load("./sample_data_and_uspat/orim_np.npy")
# or use pickle.load(open('./orim','rb'))

sensmaps=np.ones_like(orim)
sensmaps=sensmaps[:,:,np.newaxis]

# load the US pattern or make a new one
#-----------------------------------------------------
try:
     uspat=np.load('./sample_data_and_uspat/uspat_np.npy')
     # or use pickle.load(open('./uspat','rb'))
except:
     USp = US_pattern();
     uspat = USp.generate_opt_US_pattern_1D([orim.shape[0], orim.shape[1]], R=R, max_iter=100, no_of_training_profs=15)
    

# undersample the image and get the zerofilled image
#-----------------------------------------------------
usksp = UFT(orim,uspat)/np.percentile( np.abs(tUFT(UFT(orim,uspat),uspat).flatten())  ,99)
zerofilled = tUFT(usksp, uspat)

# reconstruction parameters
#-----------------------------------------------------
regtype='reg2' # which type of regularization/projection to use for the phase image. 'TV' also works well...
reg=0.1 # strength of this phase regularization. 0 means no regularization is applied
regiter = 10 # how many iterations for the phase regularization

num_iter = 302 # how many total iterations to run the reconstruction. 
dcprojiter=10 # there will be a data consistency projection every 'dcprojiter'steps
# note: this setting corresponds to 302/10 = 30 POCS iterations,
#since you do a data consistency projection every ten steps.
#the extra 2 are necessary to make sure the for loop runs until the last data
#consistency projection.

parfact = 25 # a factor for parallel computing for speeding up computations,
#i.e. doing operations in parallel for the patches, but upper bounded by memory 

# run the recon!
rec_ddp = vaerecon5.vaerecon(usksp, sensmaps=sensmaps, dcprojiter=dcprojiter, lat_dim=lat_dim, patchsize=ndims ,parfact=parfact, num_iter=num_iter, regiter=regiter, reglmb=reg, regtype=regtype)
pickle.dump(rec_ddp, open('./rec' ,'wb')   )
pickle.dump(np.abs(tFT(usksp)), open('./zerofilled' ,'wb')   )
               

# the reconstructed image is the image after the last data consistency projection
rec = rec_ddp[:,-1] # i.e. the 301th image
rec = np.reshape(rec,[orim.shape[0], orim.shape[1]])

rec_abs = np.abs(rec)
rec_phase = np.angle(rec)


# calculate RMSE while making sure the images are scaled similarly:
rmse = calc_rmse(orim , rec_abs/np.linalg.norm(rec_abs)*np.linalg.norm(orim) )   
print(rmse)      
