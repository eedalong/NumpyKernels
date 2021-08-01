import time
import numpy as np
from npkernel.kernels import long_2d_array_row_copy
Torch_Test = True

def test_array_pad():
    target_array = np.ones([230000, 768]).astype(np.int64)
    source_array = np.ones([130000, 768]).astype(np.int64)

    start_time = time.time()
    long_2d_array_row_copy(target_array, source_array)
    print(f"our kernel consumed {time.time() - start_time}")

    start_time = time.time()
    target_array[:source_array.shape[0]] = source_array
    print(f"numpy consumed {time.time() - start_time}")

    if Torch_Test:
        import torch
        target_array = torch.LongTensor(target_array)
        source_array = torch.LongTensor(source_array)
        start_time = time.time()
        target_array[: source_array.shape[0], :] = source_array
        print(f"torch consumed {time.time() - start_time}")


test_array_pad()