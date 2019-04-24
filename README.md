# Code for "MR image reconstruction using deep density priors", IEEE TMI, 2018
## K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
### tezcan@vision.ee.ethz.ch, CVL ETH Zürich


Files:

1. runrecon.py: The file that prepares and runs the reconstruction. This loads the image, creates the undersampled version and calls the reconstruction function: vaerecon5.py. After the reconstruction the reconstructed and the zero-filled images are saved as pickle files: 'rec' and 'zerofilled'. 'rec' contains the images throughout the iterations, so can be used to calculate step-wise RMSE to verify convergence.

2. vaerecon5.py: The main recon function. It contains the necessary functions for the recon, such as the multi-coil Fourier transforms, data projections, prior projections, phase projections etc... Also implements the POCS optimization scheme. The prior projection is the part that uses the VAE. The VAE is also called as a function here: definevae2.py. This function returns the necessary operation and gradients to do the prior projection.

3. definevae2.py: this function contains the VAE architecture, so runs it once to generate the graph, then loads the stored values into the variables, then rewires the graph again using the stored values, but this time with a variable (x_rec) for the input image instead of a placeholder (x_inp). This is because of some issues with an earlier version of the TF while calculating derivatives w.r.t. placeholders. Finally, this implements the necessary operations and their gradients to return to the recon function.
3.1. The weights for the trained network are are stored in 3 files by TF. These files are in the given folder. You need to change the name of the directory in the function, where the model is restored (saver.restore(xxx)), if you move them.

6. Patcher: a class to handle the patching operations. I did not want to write the patching functions in the recon function to avoid unnecessarily cumbersome code, made a class instead. You can make an instance of this at the very beginning, providing your image size and desired settings (patch size, overlap etc...) and use its two functions to convert an image to a set of patches and vice versa. Not very thoroughly tested. 

7. US_pattern.py: a class to generate US patterns. Not thoroughly tested, but should be fine if you use it as given in the main file.

Note: The shared model is trained on 790 central T1 weighted slices with 1 mm in-slice resolution (158 subjects) from the HCP dataset.

### Versions:
-------------------------

-Python 3.6.3 |Anaconda custom (64-bit)| (default, Oct  6 2017, 17:14:46)

-Tensorflow '1.9.0'

-_license                  1.1                      py36_1
-_tflow_190_select         0.0.1                       gpu    anaconda
-absl-py                   0.5.0                    py36_0    anaconda
-alabaster                 0.7.10                   py36_0  
-anaconda                  custom           py36hbbc8b67_0    anaconda
-anaconda-client           1.6.3                    py36_0  
-anaconda-navigator        1.6.2                    py36_0  
-anaconda-project          0.6.0                    py36_0  
-array2gif                 1.0.4                     <pip>
-asn1crypto                0.22.0                   py36_0  
-astor                     0.7.1                    py36_0    anaconda
-astroid                   1.4.9                    py36_0  
-astropy                   2.0.8            py36h035aef0_0    anaconda
-babel                     2.4.0                    py36_0  
-backports                 1.0                      py36_0  
-beautifulsoup4            4.6.0                    py36_0  
-bitarray                  0.8.1                    py36_0  
-blas                      1.0                         mkl  
-blaze                     0.10.1                   py36_0  
-bleach                    1.5.0                    py36_0  
-blosc                     1.14.4               hdbcaa40_0  
-bokeh                     0.12.5                   py36_1  
-boto                      2.46.1                   py36_0  
-bottleneck                1.2.1            py36h035aef0_1    anaconda
-bzip2                     1.0.6                h14c3975_5  
-cairo                     1.14.12              h77bcde2_0  
-cffi                      1.10.0                   py36_0  
-chardet                   3.0.3                    py36_0  
-click                     6.7                      py36_0  
-cloudpickle               0.2.2                    py36_0  
-clyent                    1.2.2                    py36_0  
-colorama                  0.3.9                    py36_0  
-contextlib2               0.5.5                    py36_0  
-cryptography              1.8.1                    py36_0  
-cudatoolkit               9.0                  h13b8566_0    anaconda
-cudnn                     7.1.2                 cuda9.0_0    anaconda
-cupti                     9.0.176                       0    anaconda
-curl                      7.52.1                        0  
-cycler                    0.10.0                   py36_0  
-cython                    0.25.2                   py36_0  
-cytoolz                   0.8.2                    py36_0  
-dask                      0.14.3                   py36_1  
-datashape                 0.5.4                    py36_0  
-dbus                      1.10.22              h3b5a359_0  
-decorator                 4.0.11                   py36_0  
-distributed               1.16.3                   py36_0  
-docutils                  0.13.1                   py36_0  
-entrypoints               0.2.2                    py36_1  
-et_xmlfile                1.0.1                    py36_0  
-expat                     2.1.0                         0  
-fastcache                 1.0.2                    py36_1  
-flask                     0.12.2                   py36_0  
-flask-cors                3.0.2                    py36_0  
-fontconfig                2.12.4               h88586e7_1  
-freetype                  2.8                  hab7d2ae_1  
-gast                      0.2.0                    py36_0    anaconda
-get_terminal_size         1.0.0                    py36_0  
-gevent                    1.2.1                    py36_0  
-glib                      2.53.6               h5d9569c_2  
-graphite2                 1.3.12               h23475e2_2  
-greenlet                  0.4.12                   py36_0  
-grpcio                    1.12.1           py36hdbcaa40_0    anaconda
-gst-plugins-base          1.12.4               h33fb286_0  
-gstreamer                 1.12.4               hb53b477_0  
-h5py                      2.9.0            py36h7918eee_0    anaconda
-harfbuzz                  1.7.6                hc5b324e_0  
-hdf5                      1.10.4               hb1b8bf9_0  
-heapdict                  1.0.0                    py36_1  
-html5lib                  0.999                    py36_0  
-icu                       58.2                 h9c2bf20_1  
-idna                      2.5                      py36_0  
-imageio                   2.4.1                    py36_0    anaconda
-imagesize                 0.7.1                    py36_0  
-intel-openmp              2019.0                      118  
-ipykernel                 4.6.1                    py36_0  
-ipython                   5.3.0                    py36_0  
-ipython_genutils          0.2.0                    py36_0  
-ipywidgets                6.0.0                    py36_0  
-isort                     4.2.5                    py36_0  
-itsdangerous              0.24                     py36_0  
-jbig                      2.1                           0  
-jdcal                     1.3                      py36_0  
-jedi                      0.10.2                   py36_2  
-jinja2                    2.9.6                    py36_0  
-jpeg                      9b                            0  
-jsonschema                2.6.0                    py36_0  
-jupyter                   1.0.0                    py36_3  
-jupyter_client            5.0.1                    py36_0  
-jupyter_console           5.1.0                    py36_0  
-jupyter_core              4.3.0                    py36_0  
-kiwisolver                1.0.1            py36hf484d3e_0  
-lazy-object-proxy         1.2.2                    py36_0  
-libedit                   3.1                  heed3624_0    anaconda
-libffi                    3.2.1                         1  
-libgcc                    4.8.5                         2  
-libgcc-ng                 8.2.0                hdf63c60_1    anaconda
-libgfortran               3.0.0                         1  
-libgfortran-ng            7.3.0                hdf63c60_0    anaconda
-libiconv                  1.14                          0  
-libpng                    1.6.35               hbc83047_0  
-libprotobuf               3.6.0                hdbcaa40_0    anaconda
-libsodium                 1.0.10                        0  
-libstdcxx-ng              8.2.0                hdf63c60_1    anaconda
-libtiff                   4.0.9                he85c1e1_2  
-libtool                   2.4.2                         0  
-libxcb                    1.12                          1  
-libxml2                   2.9.4                         0  
-libxslt                   1.1.29                        0  
-llvmlite                  0.25.0           py36hd408876_0    anaconda
-locket                    0.2.0                    py36_1  
-lxml                      3.7.3                    py36_0  
-lzo                       2.10                 h49e0be7_2  
-markdown                  2.6.11                   py36_0    anaconda
-markupsafe                0.23                     py36_2  
-matplotlib                2.2.2            py36h0e671d2_0  
-mistune                   0.7.4                    py36_0  
-mkl                       2018.0.3                      1  
-mkl-service               1.1.2                    py36_3  
-mkl_fft                   1.0.6            py36h7dd41cf_0  
-mkl_random                1.0.1            py36h4414c95_1  
-mpmath                    0.19                     py36_1  
-msgpack-python            0.4.8                    py36_0  
-multipledispatch          0.4.9                    py36_0  
-navigator-updater         0.1.0                    py36_0  
-nbconvert                 5.1.1                    py36_0  
-nbformat                  4.3.0                    py36_0  
-ncurses                   6.0                  h06874d7_1    anaconda
-networkx                  1.11                     py36_0  
-nibabel                   2.3.1                     <pip>
-nltk                      3.2.3                    py36_0  
-nose                      1.3.7                    py36_1  
-notebook                  5.0.0                    py36_0  
-numba                     0.40.0           py36h962f231_0    anaconda
-numexpr                   2.6.8            py36hd89afb7_0  
-numpy                     1.15.2           py36h1d66e8a_1  
-numpy-base                1.15.2           py36h81de0dd_1  
-numpydoc                  0.6.0                    py36_0  
-odo                       0.5.0                    py36_1  
-olefile                   0.44                     py36_0  
-openpyxl                  2.4.7                    py36_0  
-openssl                   1.0.2l                        0  
-packaging                 16.8                     py36_0  
-pandas                    0.23.4           py36h04863e7_0    anaconda
-pandocfilters             1.4.1                    py36_0  
-pango                     1.41.0               hd475d92_0  
-partd                     0.3.8                    py36_0  
-path.py                   10.3.1                   py36_0  
-pathlib2                  2.2.1                    py36_0  
-patsy                     0.4.1                    py36_0  
-pcre                      8.39                          1  
-pep8                      1.7.0                    py36_0  
-pexpect                   4.2.1                    py36_0  
-pickleshare               0.7.4                    py36_0  
-pillow                    5.1.0            py36h3deb7b8_0  
-pip                       9.0.1                    py36_1  
-pixman                    0.34.0                        0  
-ply                       3.10                     py36_0  
-prompt_toolkit            1.0.14                   py36_0  
-protobuf                  3.6.0            py36hf484d3e_0    anaconda
-psutil                    5.2.2                    py36_0  
-ptyprocess                0.5.1                    py36_0  
-py                        1.4.33                   py36_0  
-pycosat                   0.6.2                    py36_0  
-pycparser                 2.17                     py36_0  
-pycrypto                  2.6.1                    py36_6  
-pycurl                    7.43.0                   py36_2  
-pyflakes                  1.5.0                    py36_0  
-pygments                  2.2.0                    py36_0  
-pylint                    1.6.4                    py36_1  
-pyodbc                    4.0.16                   py36_0  
-pyopenssl                 17.0.0                   py36_0  
-pyparsing                 2.1.4                    py36_0  
-pyqt                      5.6.0                    py36_2  
-pytables                  3.4.4            py36ha205bf6_0  
-pytest                    3.0.7                    py36_0  
-python                    3.6.3                hcad60d5_0    anaconda
-python-dateutil           2.6.0                    py36_0  
-pytz                      2017.2                   py36_0  
-pywavelets                1.0.1            py36hdd07704_0    anaconda
-pyyaml                    3.12                     py36_0  
-pyzmq                     16.0.2                   py36_0  
-qt                        5.6.2               h974d657_12  
-qtawesome                 0.4.4                    py36_0  
-qtconsole                 4.3.0                    py36_0  
-qtpy                      1.2.1                    py36_0  
-readline                  7.0                  hac23ff0_3    anaconda
-requests                  2.14.2                   py36_0  
-rope                      0.9.4                    py36_1  
-ruamel_yaml               0.11.14                  py36_1  
-scikit-image              0.13.1           py36h14c3975_1    anaconda
-scikit-learn              0.20.0           py36h4989274_1  
-scipy                     1.1.0            py36hfa4b5c9_1  
-seaborn                   0.7.1                    py36_0  
-setuptools                27.2.0                   py36_0  
-simplegeneric             0.8.1                    py36_1  
-singledispatch            3.4.0.3                  py36_0  
-sip                       4.18                     py36_0  
-six                       1.10.0                   py36_0  
-snappy                    1.1.7                hbae5bb6_3  
-snowballstemmer           1.2.1                    py36_0  
-sortedcollections         0.5.3                    py36_0  
-sortedcontainers          1.5.7                    py36_0  
-sphinx                    1.5.6                    py36_0  
-spyder                    3.1.4                    py36_0  
-sqlalchemy                1.1.9                    py36_0  
-sqlite                    3.23.1               he433501_0    anaconda
-statsmodels               0.9.0            py36h035aef0_0    anaconda
-sympy                     1.0                      py36_0  
-tbb                       2019.1               hfd86e86_0    anaconda
-tblib                     1.3.2                    py36_0  
-tensorboard               1.9.0            py36hf484d3e_0    anaconda
-tensorflow                1.9.0           gpu_py36h02c5d5e_1    anaconda
-tensorflow-base           1.9.0           gpu_py36h6ecc378_0    anaconda
-tensorflow-gpu            1.9.0                hf154084_0    anaconda
-termcolor                 1.1.0                    py36_1    anaconda
-terminado                 0.6                      py36_0  
-testpath                  0.3                      py36_0  
-tk                        8.6.8                hbc83047_0    anaconda
-toolz                     0.8.2                    py36_0  
-tornado                   4.5.1                    py36_0  
-traitlets                 4.3.2                    py36_0  
-unicodecsv                0.14.1                   py36_0  
-unixodbc                  2.3.4                         0  
-wcwidth                   0.1.7                    py36_0  
-werkzeug                  0.12.2                   py36_0  
-wheel                     0.29.0                   py36_0  
-widgetsnbextension        2.0.0                    py36_0  
-wrapt                     1.10.10                  py36_0  
-xlrd                      1.0.0                    py36_0  
-xlsxwriter                0.9.6                    py36_0  
-xlwt                      1.2.0                    py36_0  
-xz                        5.2.4                h14c3975_4    anaconda
-yaml                      0.1.6                         0  
-zeromq                    4.1.5                         0  
-zict                      0.1.2                    py36_0  
-zlib                      1.2.11               hfbfcf68_1    anaconda-



