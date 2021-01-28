import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Callable, Tuple
from tqdm import tqdm
import json
import time


def _get_tensors(size: int, density: float) -> Dict[str, torch.tensor]:
    device = torch.device('cuda')
    # Create desne tensor
    dense_tensor = torch.randn((size, size), device=device)

    # Create sparse tensor based on given size and density
    nnz = int(size * size * density)
    tmp_tensor = torch.zeros(size * size, device=device)
    tmp_tensor[:nnz] = torch.randn(nnz)
    tmp_tensor = tmp_tensor[torch.randperm(size * size)]
    tmp_tensor = tmp_tensor.view(size, size)

    # Create sparse tensor from tensor
    sparse_tensor = tmp_tensor.clone().detach()
    # Create sparse_coo_tensor from tensor
    sparse_coo_tensor_from_dense = tmp_tensor.clone().detach().to_sparse()
    # Create sparse_coo_tensor from scratch
    sparse_coo_tensor = torch.sparse_coo_tensor(indices=sparse_coo_tensor_from_dense._indices(),
                                                values=sparse_coo_tensor_from_dense._values(),
                                                size=(size, size),
                                                dtype=torch.float32,
                                                device=device)
    # Create sparse_coo_tensor from scratch and coalesce
    sparse_coo_tensor_coalesce = torch.sparse_coo_tensor(indices=sparse_coo_tensor_from_dense._indices(),
                                                         values=sparse_coo_tensor_from_dense._values(),
                                                         size=(size, size),
                                                         dtype=torch.float32,
                                                         device=device).coalesce()
    # Create sparse_coo_tensor and convert it to tensor
    sparse_tensor_from_sct = torch.sparse_coo_tensor(indices=sparse_coo_tensor_from_dense._indices(),
                                                     values=sparse_coo_tensor_from_dense._values(),
                                                     size=(size, size),
                                                     dtype=torch.float32,
                                                     device=device).to_dense()
    for nnz_sct in [sparse_coo_tensor_from_dense._nnz(),
                    sparse_coo_tensor._nnz(),
                    sparse_coo_tensor_coalesce._nnz()]:
        assert abs(float(nnz) / nnz_sct - 1) < 1e-4, f"Expected nnz:{nnz}; get nnz: {nnz_sct}"

    return {'dense_tensor': dense_tensor,
            'sparse_tensor': sparse_tensor,
            'sparse_coo_tensor_from_dense': sparse_coo_tensor_from_dense,
            'sparse_coo_tensor': sparse_coo_tensor,
            'sparse_coo_tensor_coalesce': sparse_coo_tensor_coalesce,
            'sparse_tensor_from_sct': sparse_tensor_from_sct}


def _benchmark(func: Callable, t0: torch.tensor, t1: torch.tensor,
               total_time: float = 1.0, min_reps: int = 100) -> Tuple[float, float, float]:
    reps, accu_time = 0, 0
    while accu_time <= total_time or reps <= min_reps:
        t_start = time.monotonic()
        c = func(t0, t1)
        torch.cuda.synchronize()
        accu_time += (time.monotonic() - t_start)
        reps += 1
    return float(accu_time) / reps, accu_time, reps


def _test_sparse_matrix_gpu(densities: List[float] = [1, 0.1, 0.01],
                            sizes: List[int] = [128, 1024]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    dict_funcs = {'torch_mm': torch.mm,
                  'torch_matmul': torch.matmul,
                  'torch_sparse_mm': torch.sparse.mm}

    dict_dummy_ts = _get_tensors(2, 1)
    exp_results = {key_func: {key_t: defaultdict(dict) for key_t in dict_dummy_ts.keys()}
                   for key_func in dict_funcs.keys()}

    pbar = tqdm(total=len(sizes) * len(densities) * len(dict_funcs.keys()) * len(dict_dummy_ts.keys()))
    for size in sizes:
        for density in densities:
            dict_tensors = _get_tensors(size, density)
            for key_func, func in dict_funcs.items():
                for key_t, tensor in dict_tensors.items():
                    period, accu_time, reps = _benchmark(func, tensor, dict_tensors['dense_tensor'].clone().detach())
                    exp_results[key_func][key_t][size][str(density)] = period

                    pbar.update(1)
                    pbar.set_description(f'{key_func}; {key_t}; {size}; {density}; {period} = {accu_time}/ {reps}')
            del dict_tensors
            torch.cuda.empty_cache()
    pbar.close()

    filename = f'test_sparse_matrix_gpu_sizes{"-".join(list(map(str,sizes)))}.json'
    with open(filename, 'w') as f:
        print(f'Experiment result saved in {filename}')
        json.dump(exp_results, f, indent=2)

    return exp_results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        import sys
        sys.exit()

    print('=' * 8 + f'{torch.cuda.get_device_name()}' + '=' * 8)

    densities = [10**p for p in np.linspace(-4, 0, 10)]
    sizes = [128, 1024, 2048, 4096, 8192]
    exp_results = _test_sparse_matrix_gpu(densities, sizes)
