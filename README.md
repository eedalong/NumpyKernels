# NumpyKernels
Cython Kernels For Some Numpy Operations

## Array Slice
[test_script is here, check this out](tests/test_slice.py)

|Implementation|Thread|Performance|
|---|---|---|
|Numpy|1|364ms|
|Pytorch|1|225ms|
|Pytorch|\> 1|35ms|
|Our Kernel|1|440ms|
|Our Kernel|\> 1|75ms|

## Array Pad
[test_script is here, check this out](tests/test_pad.py)

|Implementation|Thread|Performance|
|---|---|---|
|Numpy|1|182ms|
|Pytorch|\> 1|45ms|
|Our Kernel|\> 1|62ms|



# Install

    python setup.py build_ext --inplace
    python setup.py install
