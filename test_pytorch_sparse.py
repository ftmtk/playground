import torch
import numpy as np
import pandas as pd
from timeit import Timer
from collections import defaultdict

from typing import List
from tqdm import tqdm

import time


def _get_adjacent_matrix_rand(density: float = 0.1, nrows: int = 2048, ncols: int = 2048, dtype: torch.dtype = torch.int32):
    nedges = int(np.ceil(density * nrows * ncols))
    # Initialize a new adjacent matrix
    adjmat = np.zeros(nrows * ncols)
    locs = np.random.choice(len(adjmat), size=nedges, replace=False)
    adjmat[locs] = 1

    return torch.as_tensor(adjmat.reshape(nrows, ncols), dtype=dtype)


def _test_sparse_matrix_gpu(densities: List[float] = [1, 0.1, 0.01], sizes: List[int] = [128, 1024]):
    def _benchmark(func, t0, t1, total_time=1.0):
        reps, accu_time = 0, 0
        while accu_time <= total_time:
            t_start = time.monotonic()
            c = func(t0, t1)
            torch.cuda.synchronize()
            accu_time += (time.monotonic() - t_start)
            reps += 1
        print(accu_time, reps)
        return float(accu_time) / reps

    def _get_tensors(size, density):
        device = torch.device('cuda')
        n = size
        torch_dm = torch.randn((n, n), device=device)
        # TODO: Matrix setting not right
        torch_sparse_dm = torch.randn((n, n), device=device)
        torch_sparse_dm[torch.rand_like(torch_dm) > density] = 0
        torch_sparse_sm_to_sparse = torch_sparse_dm.to_sparse()
        torch.cuda.empty_cache()
        torch_sparse_sm = torch.sparse_coo_tensor(torch_sparse_sm_to_sparse._indices(),
                                                  torch_sparse_sm_to_sparse._values(),
                                                  size=(n, n))
        torch_sparse_sm_coalesce = torch.sparse_coo_tensor(torch_sparse_sm_to_sparse._indices(),
                                                           torch_sparse_sm_to_sparse._values(),
                                                           size=(n, n)).coalesce()
        return {'torch_dm': torch_dm,
                'torch_sparse_dm': torch_sparse_dm,
                'torch_sparse_sm_to_sparse': torch_sparse_sm_to_sparse,
                'torch_sparse_sm': torch_sparse_sm,
                'torch_sparse_sm_coalesce': torch_sparse_sm_coalesce}

    dict_funcs = {'torch_mm': torch.mm,
                  'torch_matmul': torch.matmul,
                  'torch_sparse_mm': torch.sparse.mm}

    dict_dummy_ts = _get_tensors(2, 1)
    exp_results = {key_func: {key_t: defaultdict(dict) for key_t in dict_dummy_ts.keys()}
                   for key_func in dict_funcs.keys()}
    # result = {key: defaultdict(defaultdict(dict)) for key in dict_funcs.keys()}

    pbar = tqdm(total=len(sizes) * len(densities) * len(dict_funcs.keys()) * 4)
    for size in sizes:
        for density in densities:
            dict_tensors = _get_tensors(size, density)
            for key_func, func in dict_funcs.items():
                for key_t, tensor in dict_tensors.items():
                    if key_func == 'torch_sparse_mm' and 'sparse' not in key_t:
                        continue
                    period = _benchmark(func, tensor, dict_tensors['torch_dm'].detach())
                    exp_results[key_func][key_t][size][str(density)] = period

                    pbar.update(1)
                    pbar.set_description(f'{key_func}; {key_t}; {size}; {density}; {period}')
    pbar.close()
    return exp_results


def _test_sparse_matrix(densities: List[float] = [1, 0.1, 0.01],
                        sizes: List[int] = [128, 1024, 8192],
                        use_cuda: bool = True):
    def _get_setup_str(size: int, density: float,
                       tensorA_name: str = 'adjmat', tensorB_name: str = 'dense_matrix', use_cuda: bool = True):
        setup_str = ["import torch",
                     "import numpy as np",
                     f"device = torch.device('{'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'}')",
                     f"size={size}",
                     f"{tensorB_name}_1 = torch.rand(size=[{size}, {size}], dtype=torch.float32).to(device)",
                     f"{tensorB_name}_2 = torch.rand(size=[{size}, {size}], dtype=torch.float32).to(device)"]
        tensor_A_str = [f"nnz = int({np.ceil(density*size*size)})",
                        f"adjmat = np.zeros({size*size})",
                        f"locs = np.random.choice(len(adjmat), size=nnz, replace=False)",
                        f"adjmat[locs] = 1",
                        f"sparse_{tensorA_name} = torch.sparse_coo_tensor(indices = torch.tensor([torch.randint(size,(nnz,)).tolist(),torch.randint(size,(nnz,)).tolist()]), values = torch.randn(nnz), size=[size,size], dtype=torch.float32, device=device)",
                        f"{tensorA_name} = sparse_{tensorA_name}.to_dense()",
                        # f"{tensorA_name} = torch.as_tensor(adjmat.reshape({size}, {size}), dtype=torch.float32).to(device)",
                        # f"sparse_{tensorA_name} = {tensorA_name}.to_sparse()"
                        ]

        return ';'.join(setup_str + tensor_A_str)

    def _get_test_exec_and_setup(size, density, use_cuda: bool = True):
        setup_str = _get_setup_str(size, density, 'adjmat', 'dense_matrix', use_cuda)
        dic = {'torch_matmul_dense': ['torch.matmul(dense_matrix_1, dense_matrix_2)', setup_str],
               'torch_mm': ['torch.mm(adjmat, dense_matrix_1)', setup_str],
               'torch_matmul': ['torch.matmul(adjmat, dense_matrix_1)', setup_str],
               'torch_matmul_rev': ['torch.matmul(dense_matrix_1, adjmat)', setup_str],
               'tensor_matmul': ['adjmat.matmul(dense_matrix_1)', setup_str],
               'tensor_matmul_rev': ['dense_matrix_1.matmul(adjmat)', setup_str],
               'torch_einsum': ["torch.einsum('ik,kj->ij', adjmat, dense_matrix_1)", setup_str],
               'torch_einsum_rev': ["torch.einsum('ik,kj->ij', dense_matrix_1, adjmat)", setup_str],
               'torch_sparse_mm': ["torch.sparse.mm(sparse_adjmat, dense_matrix_1)", setup_str]
               }
        return dic

    dict_dummy = _get_test_exec_and_setup(2, 0.5)
    exp_results = {key: defaultdict(dict) for key in dict_dummy.keys()}
    result = {key: defaultdict(dict) for key in dict_dummy.keys()}

    pbar = tqdm(total=len(sizes) * len(densities) * len(dict_dummy.keys()))
    for n in sizes:
        dict_exec_setup = _get_test_exec_and_setup(n, 1, use_cuda)
        key = 'torch_matmul_dense'
        exec_str, setup_str = dict_exec_setup[key]
        reps, time = Timer(stmt=exec_str, setup=setup_str).autorange()
        period = float(time) / reps * 1e5
        exp_results[key][n]['1'] = period
        pbar.set_description(f"{key}: {n} x {n}; {1.00} done. Takes {time:.4f} with {reps} reps")
        pbar.update(1)

    for n in sizes:
        for density in densities:
            dict_exec_setup = _get_test_exec_and_setup(n, density, use_cuda)
            for key, (exec_str, setup_str) in dict_exec_setup.items():
                if key == 'torch_matmul_dense':
                    continue

                reps, time = Timer(stmt=exec_str, setup=setup_str).autorange()
                period = float(time) / reps * 1e5

                exp_results[key][n][str(density)] = period
                pbar.set_description(f"{key}: {n} x {n}; {density} done. Takes {time:.4f} with {reps} reps")
                pbar.update(1)

            for key, (exec_str, setup_str) in dict_exec_setup.items():
                if key != 'torch_matmul_dense':
                    result[key][n][str(density)] = (exp_results['torch_matmul_dense'][n]['1'] /
                                                    exp_results[key][n][str(density)] - 1) * 100
    pbar.close()
    return exp_results, result


if __name__ == "__main__":
    for use_cuda in [True]:
        tag = f'{torch.cuda.get_device_name()}' if use_cuda and torch.cuda.is_available() else 'cpu'
        print('=' * 8 + f'{tag}' + '=' * 8)

        # densities = [1, 0.5, 0.1, 0.05, 0.025, 0.015, 0.01, 0.005, 0.001, 1e-4]
        # densities = [1, 0.1, 0.01]
        # densities = [10**p for p in np.linspace(-4, 0, 10)]
        # sizes = [128, 1024, 4096]
        densities = [0.1, 0.01]
        sizes = [8192]

        exp_results = _test_sparse_matrix_gpu(densities, sizes)
        for key_func, func_itms in exp_results.items():
            print('=' * 8 + f' {key_func}')
            for key_t, t_itms in func_itms.items():
                print('.' * 16 + f' {key_t}')
                df = pd.DataFrame.from_dict(t_itms)
                print(df.to_string())

                df.to_csv(f'Syncronize_{key_func}_{key_t}.csv')

        """
        exp_results, result = _test_sparse_matrix(densities, sizes, use_cuda)

        dfs = {key: pd.DataFrame.from_dict(dic) for key, dic in exp_results.items()}
        dfs_comp = {key: pd.DataFrame.from_dict(dic) for key, dic in result.items()}
        for key, df in dfs.items():
            print(f'======== {key}')
            print(df.to_string())

            df.to_csv(f"{key}_{tag}.csv")
            if key != 'torch_matmul_dense':
                dfs_comp[key].to_csv(f'Comp_{key}_vs_torch_matmul_{tag}.csv')
                print(f"." * 16 + " faster than torch_matmul_dense by")
                print(dfs_comp[key].to_string())
        """
