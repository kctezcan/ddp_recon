# K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
# code for reproducing "MR image reconstruction using deep density priors"
# IEEE TMI, 2018
# tezcan@vision.ee.ethz.ch, CVL ETH Zürich

import numpy as np
import os
import pickle
import vaerecon
from US_pattern import US_pattern
import h5py

# to tell tensorflow which GPU to use
#-----------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())
print( os.environ['SGE_GPU'])

#define the necessary encoding operations and their transposes
#-----------------------------------------------------
def FT (x):
     #inp: [nx, ny]
     #out: [nx, ny, ns]
     return np.fft.fftshift(    np.fft.fft2( sensmaps*np.tile(x[:,:,np.newaxis],[1,1,sensmaps.shape[2]]) , axes=(0,1)  ),   axes=(0,1)    )

def tFT (x):
     #inp: [nx, ny, ns]
     #out: [nx, ny]
     
     temp = np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )
     return np.sum( temp*np.conjugate(sensmaps) , axis=2)  / np.sum(sensmaps*np.conjugate(sensmaps),axis=2)


def UFT(x, uspat):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny, ns]
     
     return np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])*FT(x)

def tUFT(x, uspat):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny]
     
     tmp1 = np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])
     print(x.shape)
     print(tmp1.shape)
     
     return  tFT( tmp1*x )
     

# RMSE calculation
#-----------------------------------------------------
def calc_rmse(rec,imorig):
     return 100*np.sqrt(np.sum(np.square(np.abs(rec) - np.abs(imorig))) / np.sum(np.square(np.abs(imorig))) )

#which VAE model to use
#-----------------------------------------------------   
ndims=28
lat_dim=60



# the US factor
#-----------------------------------------------------
R=2

#load the test image and the coil maps
#-----------------------------------------------------        
#unfortunately, due to complications in saving complex valued data, we save
#and load the complex and real parts seperately
f = h5py.File('./sample_data_and_uspat/acq_im_real.h5', 'r')
imr = np.array((f['DS1']))
f = h5py.File('./sample_data_and_uspat/acq_im_imag.h5', 'r')
imi = np.array((f['DS1']))
orim = np.rot90(np.transpose(imr+1j*imi),3)

#get the k-space data
ksp = np.fft.ifftn(np.fft.fftshift(np.fft.fftn(orim,axes=[0,1]),axes=[0,1]),axes=[0,1])


#again we save and load the complex and real parts seperately for coil maps
f = h5py.File('./sample_data_and_uspat/acq_im_real.h5', 'r')
espsr = np.array((f['DS1']))
f = h5py.File('./sample_data_and_uspat/acq_im_imag.h5', 'r')
espsi = np.array((f['DS1']))

esps= np.rot90(np.transpose(espsr+1j*espsi),3)
sensmaps = esps.copy()
     
#rotate images for canonical orientation
sensmaps=np.rot90(np.rot90(sensmaps))
orim=np.rot90(np.rot90(orim))

#normalize the espirit coil maps
sensmaps=sensmaps/np.tile(np.sum(sensmaps*np.conjugate(sensmaps),axis=2)[:,:,np.newaxis],[1, 1, sensmaps.shape[2]])

#get the coil combined image
ddimc = tFT(ksp)

#load the undersampling pattern
patt = pickle.load(open('./sample_data_and_uspat/acq_im_us_patt_R2','rb'))

#make the undersampled kspace
usksp = ksp * np.tile(patt[:,:,np.newaxis],[1, 1, ksp.shape[2]])

# normalize the kspace
usksp = usksp / np.percentile(  np.abs(tUFT(usksp, patt)).flatten()   ,99)

#=============================================================================
onlydciter=10 # do 10 only SENSE iterations, then switch on the prior projections
num_iter = 22 # total number of iterations

regtype='reg2' # dummy choice here, switched off anyways
reg=0 # do not use any phase projection, since the SENSE recon takes care of the phase sufficiently

rec_vae = vaerecon.vaerecon(usksp, sensmaps=sensmaps, dcprojiter=10, onlydciter=onlydciter, lat_dim=lat_dim, patchsize=ndims, parfact=20, num_iter=num_iter, regiter=15, reglmb=reg, regtype=regtype)
pickle.dump(rec_vae, open('./rec_ddp_espirit' ,'wb')   )
          
          
