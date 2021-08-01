# NumpyKernels
Cython Kernels For Some Numpy Operations. 

最近工作上有很多需要用到Numpy的地方，Profiling之后发现Numpy的访存操作真的太慢了，严重影响了我们的系统性能，于是我用Cython写了一些Kernel来加速这些访存操作，
如下是Kernel的Profling结果。可以发现Pytroch Tensor实现真的很高效，以及我们的kernel相比较Numpy的原生操作也的确快不少。

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

*Case 1*

|Implementation|Thread|Performance|
|---|---|---|
|Numpy|1|182ms|
|Pytorch|\> 1|45ms|
|Our Kernel|\> 1|62ms|

*Case 2*

|Implementation|Thread|Performance|
|---|---|---|
|Numpy|1|261ms|
|Pytorch|\> 1|42ms|
|Our Kernel|\> 1|54ms|

## Comming Soon

-[ ] Parallel Graph Sampling Kernel In Cython


# Install

    // python setup.py build_ext --inplace
    python setup.py install
