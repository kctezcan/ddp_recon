# Code for "MR image reconstruction using deep density priors", IEEE TMI, 2018
## K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
### tezcan@vision.ee.ethz.ch, CVL ETH Zürich


Files:

1. runrecon.py: The file that prepares and runs the reconstruction for the sample image from the HCP dataset. This loads the image, creates the undersampled version and calls the reconstruction function: vaerecon.py. After the reconstruction the reconstructed and the zero-filled images are saved as pickle files: 'rec' and 'zerofilled'. 'rec' contains the images throughout the iterations, so can be used to calculate step-wise RMSE to verify convergence.

2. runrecon_acquired_image.py: The file that prepares and runs the reconstruction for an acquired image. This loads the image, the pre-calculated ESPIRiT coil maps, creates the undersampled version and calls the reconstruction function: vaerecon.py. After the reconstruction the reconstructed and the zero-filled images are saved as pickle files: 'rec_ddp_espirit'. 'rec_ddp_espirit' contains the images throughout the iterations, so can be used to calculate step-wise RMSE to verify convergence.

3. vaerecon.py: The main recon function. It contains the necessary functions for the recon, such as the multi-coil Fourier transforms, data projections, prior projections, phase projections etc... Also implements the POCS optimization scheme. The prior projection is the part that uses the VAE. The VAE is also called as a function here: definevae2.py. This function returns the necessary operation and gradients to do the prior projection.

4. definevae.py: This function contains the VAE architecture, so runs it once to generate the graph, then loads the stored values into the variables, then rewires the graph again using the stored values, but this time with a variable (x_rec) for the input image instead of a placeholder (x_inp). This is because of some issues with an earlier version of the TF while calculating derivatives w.r.t. placeholders. Finally, this implements the necessary operations and their gradients to return to the recon function.

5. Patcher: A class to handle the patching operations. I did not want to write the patching functions in the recon function to avoid unnecessarily cumbersome code, made a class instead. You can make an instance of this at the very beginning, providing your image size and desired settings (patch size, overlap etc...) and use its two functions to convert an image to a set of patches and vice versa. Not very thoroughly tested. 

6. US_pattern.py: A class to generate US patterns. Not thoroughly tested, but should be fine if you use it as given in the main file.

7. trained_model: The shared model is trained on 790 central T1 weighted slices with 1 mm in-slice resolution (158 subjects) from the HCP dataset.

8. sample_data_and_uspat: We share a sample image from the HCP dataset (single coil, no phase) and an image from a volunteer acquired for this study (16 coils, complex image) along with the corresponding ESPIRiT coil maps. We provide the images saved as two separate files, for the complex and imaginary parts due to complications with saving complex numbers. We also provide sample undersampling patterns for both images for R=2.

Note: The dependencies are given in the requirements.txt. You can use "$ conda create --name <env> --file <this file>" to create an environmet with the given dependencies.


