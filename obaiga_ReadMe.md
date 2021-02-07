## Installation

1. Anaconda version: Anaconda3-2020.02-Windows-x86_64

Make sure that the installer adds Anaconda to the system PATH  
Make sure that the installer registers it as your default Python

If you already have an Anaconda. The installation path could be [the current Anaconda path] + [\envs], e.g. C:\Users\95316\Anaconda2\envs

2. Requirement packages
```python
pip install numpy
pip install scipy
pip install Pillow
pip install cython
pip install matplotlib
pip install scikit-image
pip install tensorflow==1.14.0
pip install keras==2.2.0
pip install opencv-python
pip install h5py
pip install imgaug
pip install IPython

## It may report "module 'keras_applications' has no attribute 'set_keras_submodules'"
pip install keras_applications==1.0.7

## check tensorflow & keras version 
import tensorflow
print(tensorflow.__version__)

## To ensure the program running on GPU
conda install tensorflow-gpu

## Check whether GPU is being or not
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

pip install keras_applications==1.0.7
## double check tensorflow & keras version 

```
