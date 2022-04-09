## Installation

1. Anaconda version: Anaconda3-2020.02-Windows-x86_64

Make sure that the installer adds Anaconda to the system PATH  
Make sure that the installer registers it as your default Python

If you already have an Anaconda. The installation path could be [the current Anaconda path] + [\envs], e.g. C:\Users\95316\Anaconda2\envs

2. Requirement packages
```
conda create -n maskrcnn python=3.7.6
conda activate maskrcnn

pip install numpy
pip install scipy
pip install Pillow
pip install cython
pip install matplotlib
pip install scikit-image
pip install opencv-python
pip install imgaug
pip install IPython
```
```
conda install tensorflow==1.14.0
conda install keras
conda install tensorflow-gpu  ## for wins
conda install h5py==2.10.0
pip install keras_applications==1.0.7
```

``` python
python
import tensorflow as tf
import keras
import h5py

print(tf.__version__)
print(keras.__version__)
print(h5py.__version__)

## Check whether GPU is being or not
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
show
```
1.14.1
2.3.1
2.10.0

[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 6757537999743423464
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 7109751604
locality {
  bus_id: 1
  links {
  }
}
incarnation: 6561882754128707177
physical_device_desc: "device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1"
]
```
## Issues
1. Description:Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above. #24828

Alternative description: Delete the underlying status object from memory otherwise it stays alives there is a reference to status from this from the traceback due to UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above. [[{{node sequential_1_1/Conv1/convolution}}]] [[{{node loss/add_7}}]]

Solution: Restart computer
