# K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
# code for reproducing "MR image reconstruction using deep density priors"
# IEEE TMI, 2018
# tezcan@vision.ee.ethz.ch, CVL ETH Zürich

import numpy as np
import nibabel as nib
import scipy.misc as smi
from US_pattern import US_pattern

from skimage.util.shape import view_as_blocks


class Patcher:
     # Class that creates patches of images
     
     #you need to make sure that the image size is a multiple of patchsize
     #but you can have a step size, where you do not get full patches.
     
     #====================================================================
     def __init__(self, imsize, patchsize, step, nopartials, contatedges):
          self.patchsize = patchsize
          self.step = step
          self.imsize=imsize
          self.imsizeorig=np.array(imsize)
          self.nopartials = nopartials
          self.contatedges = contatedges
          
          self.diviim =[]
          self.genpatchsizes=[]
          self.noOfPatches=0
          
          
          
          #if you want to be able to use patchsizes not dividor of image size
          if self.contatedges:
               if (self.imsize == (np.ceil(self.imsizeorig/self.patchsize)*self.patchsize).astype(int)).all():
                    self.contatedges=False
               else:
                    self.imsize = (np.ceil(self.imsizeorig/self.patchsize)*self.patchsize).astype(int)
               
          self.getDivImage()
          
          
          
     def im2patches(self,img):
          
          #pad images with srap to make the image size multiple of patchsize
          if self.contatedges:
               sd=self.imsize - self.imsizeorig
               img = np.pad(img,[ (0,sd[0]), (0,sd[1])  ], mode='wrap'  )
          
          ptchs=[]
          
          for ix in range(0,self.imsize[0],self.step):
               for iy in range(0,self.imsize[1],self.step):
                    
                    ptc = img[ix:ix+self.patchsize, iy:iy+self.patchsize]
                    
                    if ((ptc.shape[0] != self.patchsize) or (ptc.shape[1] != self.patchsize)) and self.nopartials:
                         pass
                    else:  
                         ptchs.append(ptc)
               
          return ptchs
          
          
          
     def patches2im(self,patches, combsq=False):
          
          if len( self.diviim):
               pass
          else:
               self.getDivImage()
              
          if self.contatedges:     
               tmp=np.zeros(self.imsize, dtype=np.complex128)
          else:
               tmp=np.zeros(self.imsizeorig, dtype=np.complex128)
               
          ctr=0
          
          for ix in range(0,self.imsize[0],self.step):
               for iy in range(0,self.imsize[1],self.step):
                    
                    tt=tmp[ix:ix+self.patchsize, iy:iy+self.patchsize]
                    
                    
                    if ((tt.shape[0] != self.patchsize) or (tt.shape[1] != self.patchsize)) and self.nopartials:
                         pass
                    else:
                         tmp[ix:ix+self.patchsize, iy:iy+self.patchsize] = tmp[ix:ix+self.patchsize, iy:iy+self.patchsize] + patches[ctr]
                         ctr=ctr+1
          
          if not combsq:
               tmp=tmp/self.diviim
          else:
               tmp=tmp/np.square(self.diviim)
               
          tmp=tmp[0:self.imsizeorig[0], 0:self.imsizeorig[1]]
          
          return tmp
          
          
     def getDivImage(self):
          
          if self.contatedges:
               tmp=np.zeros(self.imsize)
          else:
               tmp=np.zeros(self.imsizeorig)
               
          gensizes=[]
          
          for ix in range(0,self.imsize[0],self.step):
               for iy in range(0,self.imsize[1],self.step):
                    tt=tmp[ix:ix+self.patchsize, iy:iy+self.patchsize] 
                    
                    if ((tt.shape[0] != self.patchsize) or (tt.shape[1] != self.patchsize)) and self.nopartials:
                         pass
                    else: 
                         gensizes.append(tt.shape)#keep this to check patch sizes later
                         tmp[ix:ix+self.patchsize, iy:iy+self.patchsize] = tmp[ix:ix+self.patchsize, iy:iy+self.patchsize] + 1
          
          if (tmp==0).any():
               print("KCT-WARNING: the selected patching scheme does not allow covering of all the image! Some pixels are not in any of the patches.")
               
          tmp[np.where(tmp==0)]=1 #do as if full coverage were provided anyways...
             
          self.diviim = tmp
          self.genpatchsizes = gensizes
          self.noOfPatches = len(gensizes)
          
          
                    
          
               
               
                              
               

