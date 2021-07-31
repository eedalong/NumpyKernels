import time
import numpy as np
from npkernel.kernels import long_2d_array_slice
def test_array_slice():
    input_array = np.ones([230000, 768]).astype(np.int64)
    indices = (np.arange(140000)).astype(np.int64)

    np.random.shuffle(indices)
    start_time = time.time()
    res = long_2d_array_slice(input_array, indices)
    print(f"kernel consumed {time.time() - start_time}, res shape {res.shape}")

    start_time = time.time()
    res = input_array[indices]
    print(f"numpy consumed {time.time() - start_time}, res shape {res.shape}")
    
    import torch
    array_tensor = torch.Tensor(input_array)
    indice_tensor = torch.LongTensor(indices)

    
    start_time = time.time()
    res = array_tensor[indice_tensor]
    print(f"torch consumed {time.time() - start_time}, res shape {res.shape}")

test_array_slice()
