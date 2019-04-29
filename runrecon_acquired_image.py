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
kspr = np.array((f['DS1']))
f = h5py.File('./sample_data_and_uspat/acq_im_imag.h5', 'r')
kspi = np.array((f['DS1']))
ksp = np.rot90(np.transpose(kspr+1j*kspi),3)

#get the k-space data
ksp = np.fft.ifftn(np.fft.fftshift(np.fft.fftn(ksp,axes=[0,1]),axes=[0,1]),axes=[0,1])


#again we save and load the complex and real parts seperately for coil maps
f = h5py.File('./sample_data_and_uspat/acq_coilmaps_espirit_real.h5', 'r')
espsr = np.array((f['DS1']))
f = h5py.File('./sample_data_and_uspat/acq_coilmaps_espirit_imag.h5', 'r')
espsi = np.array((f['DS1']))

esps= np.rot90(np.transpose(espsr+1j*espsi),3)
sensmaps = esps.copy()
     
#rotate images for canonical orientation
sensmaps=np.rot90(np.rot90(sensmaps))
ksp=np.rot90(np.rot90(ksp))

#normalize the espirit coil maps
sensmaps=sensmaps/np.tile(np.sum(sensmaps*np.conjugate(sensmaps),axis=2)[:,:,np.newaxis],[1, 1, sensmaps.shape[2]])

#load the undersampling pattern
patt = pickle.load(open('./sample_data_and_uspat/acq_im_us_patt_R2','rb'))

#make the undersampled kspace
usksp = ksp * np.tile(patt[:,:,np.newaxis],[1, 1, ksp.shape[2]])

# normalize the kspace
tmp = tFT(usksp)
usksp=usksp/np.percentile(  np.abs(tmp).flatten()   ,99)

#=============================================================================
onlydciter=10 # do 10 only SENSE iterations, then switch on the prior projections

num_pocs_iter = 10 # number of total POCS iterations
dcprojiter=10 # there will be a data consistency projection every 'dcprojiter'steps

num_iter = num_pocs_iter*dcprojiter+2 # how many total iterations to run the reconstruction. 
#notice you need to take num_iter some multiple of dcprojiter + 2, so that the data consistency
#projection runs as the last step.

regtype='reg2' # dummy choice here, switched off anyways
reg=0 # do not use any phase projection, since the SENSE recon takes care of the phase sufficiently


rec_ddp = vaerecon.vaerecon(usksp, sensmaps=sensmaps, dcprojiter=dcprojiter, onlydciter=onlydciter, lat_dim=lat_dim, patchsize=ndims, parfact=20, num_iter=num_iter, regiter=15, reglmb=reg, regtype=regtype)
pickle.dump(rec_ddp, open('./rec_ddp_espirit' ,'wb')   )

# calculate RMSE
orim = tFT(ksp) # the fully sampled image

rmses=np.zeros(num_iter-1)
for ix in range(num_iter-1):
    rec = np.reshape(rec_ddp[:,ix],[orim.shape[0], orim.shape[1]])
    rmses[ix] = calc_rmse(orim, rec/np.linalg.norm(rec)*np.linalg.norm(orim))

#print or plot the rmses values per iteration
print(rmses)
          
          
