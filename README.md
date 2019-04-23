# Code for "MR image reconstruction using deep density priors", IEEE TMI, 2018
## K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
### tezcan@vision.ee.ethz.ch, CVL ETH Zürich


Some explanation since the code can be confusing.

1. runrecon.py: The file that prepares and runs the reconstruction. This loads the image, creates the undersampled version and calls the reconstruction function: vaerecon5.py. After the reconstruction the reconstructed and the zero-filled images are saved as pickle files: 'rec' and 'zerofilled'.

2. vaerecon5.py: The main recon function. It contains the necessary functions for the recon, such as the multi-coil Fourier transorms, data projections, prior projections, phase projections etc... Also implements the POCS optimization scheme. The prior projection is the part that uses the VAE. The VAE is also called as a function here: definevae2.py. This function returns the necessary operation and gradients to do the prior projection.

3. definevae2.py: this function contains the VAE architecture, so runs it once to generate the graph, then loads the stored values into the variables, then rewires the graph again using the stored values, but this time with a variable (x_rec) for the input image instead of a placeholder (x_inp). This is because of some issues with an earlier version of the TF while calculating derivatives w.r.t. placeholders. Finally, this implements the necessary operations and their gradients to return to the recon function.
3.1. The weights for the trained network are are stored in 3 files by TF. These files are in the given folder. You need to change the name of the directory in the function, where the model is restored (saver.restore(xxx)), if you move them.

6. Patcher: a class to handle the patching operations. I did not want to write the patching functions in the recon function to avoid unnecessarily cumbersome code, made a class instead. You can make an instance of this at the very beginning, providing your image size and desired settings (patch size, overlap etc...) and use its two functions to convert an image to a set of patches and vice versa. Not very thoroughly tested. 

7. US_pattern.py: a class to generate US patterns. Not thoroughly tested, but should be fine if you use it as given in the main file.

Note: The model is trained on 790 central slices with 1 mm in-slice resolution (158 subjects) from the HCP dataset.



