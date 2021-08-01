import time
import numpy as np
from npkernel.kernels import long_2d_array_row_copy, long_2d_array_col_copy
Torch_Test = True

def test_array_col_pad():
    
    print(f"PARALLEL OPERATION IS GOOD WHEN COLUMN IS LARGE")
    target_array = np.ones([230000, 768]).astype(np.int64)
    source_array = np.ones([130000, 768]).astype(np.int64)

    start_time = time.time()
    long_2d_array_col_copy(target_array, source_array)
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

def test_array_row_pad():
    print(f"PARALLEL OPERATION IS GOOD WHEN ROW IS LARGE")
    target_array = np.ones([1000, 1024 * 250]).astype(np.int64)
    source_array = np.ones([1000, 130000]).astype(np.int64)

    start_time = time.time()
    long_2d_array_row_copy(target_array, source_array)
    print(f"our kernel consumed {time.time() - start_time}")

    start_time = time.time()
    target_array[:, :source_array.shape[1]] = source_array
    print(f"numpy consumed {time.time() - start_time}")

    if Torch_Test:
        import torch
        target_array = torch.LongTensor(target_array)
        source_array = torch.LongTensor(source_array)
        start_time = time.time()
        target_array[:, :source_array.shape[1]] = source_array
        print(f"torch consumed {time.time() - start_time}")

def test_array_row_pad_bad():
    print(f"PARALLEL OPERATION IS BAD WHEN ROW IS SMALL")
    target_array = np.ones([2, 1024 * 250]).astype(np.int64)
    source_array = np.ones([2, 130000]).astype(np.int64)

    start_time = time.time()
    long_2d_array_row_copy(target_array, source_array)
    print(f"our kernel consumed {time.time() - start_time}")

    start_time = time.time()
    target_array[:, :source_array.shape[1]] = source_array
    print(f"numpy consumed {time.time() - start_time}")

    if Torch_Test:
        import torch
        target_array = torch.LongTensor(target_array)
        source_array = torch.LongTensor(source_array)
        start_time = time.time()
        target_array[:, :source_array.shape[1]] = source_array
        print(f"torch consumed {time.time() - start_time}")



test_array_col_pad()
print("#" * 60)
test_array_row_pad()
print("#" * 60)
test_array_row_pad_bad()
