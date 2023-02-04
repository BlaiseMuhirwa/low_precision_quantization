
# Low-Precision Quantizer

`lpq`(Low-Precision Quantizer) is a method for performing integer
quantization. It is implemented in C++ with complete Python bindings. 
In order for the Python wheel to work, make sure you have python 3.7 and
above installed. 

## Building the Package 
In order to build this package, first clone this repository with the 
available submodules

```shell
$ git clone https://github.com/BlaiseMuhirwa/low_precision_quantization.git --recurse-submodules
```

Open the `low_precision_quantization` folder and install the necessary 
Python libraries via `pip`. I recommend first creating a virtual environment.

```shell 
$ python3 -m venv lpq_venv
$ source lpq_venv/bin/activate
$(lpq_venv) pip install -r requirements.txt 

```

Now to build the package, run the setup file, which will invoke `cmake` in order
to create a wheel file for the package. 
```shell
$(lpq_venv) python setup.py bdist_wheel 
```

This will create a wheel file in a `dist` folder. The file should be named something
like `dist/lpq-0.0.1-cp310-cp310-macosx_12_0_arm64.whl`. The final step is to just
install the library with pip. 

```shell
$(lpq_venv) pip install dist/dist/lpq-0.0.1-cp310-cp310-macosx_12_0_arm64.whl

```

You can then import this library in a python script as follows:

```shell 
from lpq import quantizer

quantizer = quantizer.LowPrecisionQuantizerInt8()
help(quantizer)

```