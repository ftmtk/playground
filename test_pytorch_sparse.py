import torch
import numpy as np
import pandas as pd
from timeit import Timer
from collections import defaultdict

from typing import List
from tqdm import tqdm


def _get_adjacent_matrix_rand(density: float = 0.1, nrows: int = 2048, ncols: int = 2048, dtype: torch.dtype = torch.int32):
    nedges = int(np.ceil(density * nrows * ncols))
    # Initialize a new adjacent matrix
    adjmat = np.zeros(nrows * ncols)
    locs = np.random.choice(len(adjmat), size=nedges, replace=False)
    adjmat[locs] = 1

    return torch.as_tensor(adjmat.reshape(nrows, ncols), dtype=dtype)


def _test_sparse_matrix(number: int = 10, densities: List[float] = [1, 0.1, 0.01], use_cuda: bool = True):
    def _get_setup_str(size: int, density: float,
                       tensorA_name: str = 'adjmat', tensorB_name: str = 'dense_matrix', use_cuda: bool = True):
        setup_str = ["import torch",
                     "import numpy as np",
                     f"device = torch.device('{'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'}')",
                     f"{tensorB_name} = torch.rand(size=[{size}, {size}], dtype=torch.float32).to(device)"]
        tensor_A_str = [f"nedges = int(np.ceil({density} * {size} * {size}))",
                        f"adjmat = np.zeros({size} * {size})",
                        f"locs = np.random.choice(len(adjmat), size=nedges, replace=False)",
                        f"adjmat[locs] = 1",
                        f"{tensorA_name} = torch.as_tensor(adjmat.reshape({size}, {size}), dtype=torch.float32).to(device)",
                        f"sparse_{tensorA_name} = {tensorA_name}.to_sparse()"]

        return ';'.join(setup_str + tensor_A_str)

    def _get_test_exec_and_setup(size, density, use_cuda: bool = True):
        setup_str = _get_setup_str(size, density, 'adjmat', 'dense_matrix', use_cuda)
        dic = {'torch_matmul': ['torch.matmul(adjmat, dense_matrix)', setup_str],
               'torch_matmul_rev': ['torch.matmul(dense_matrix, adjmat)', setup_str],
               'tensor_matmul': ['adjmat.matmul(dense_matrix)', setup_str],
               'tensor_matmul_rev': ['dense_matrix.matmul(adjmat)', setup_str],
               'torch_einsum': ["torch.einsum('ik,kj->ij', adjmat, dense_matrix)", setup_str],
               'torch_einsum_rev': ["torch.einsum('ik,kj->ij', dense_matrix, adjmat)", setup_str],
               'torch_sparse_mm': ["torch.sparse.mm(sparse_adjmat, dense_matrix)", setup_str]
               }
        return dic

    dict_dummy = _get_test_exec_and_setup(2, 0.5)
    exp_results = {key: defaultdict(dict) for key in dict_dummy.keys()}
    result = {key: defaultdict(dict) for key in dict_dummy.keys()}

    from_dim, to_dim = 8, 14
    pbar = tqdm(total=(to_dim - from_dim + 1) * len(densities) * len(dict_dummy.keys()))
    for p in range(from_dim, to_dim):
        n = 2**p
        for density in densities:

            dict_exec_setup = _get_test_exec_and_setup(n, density, use_cuda)
            for key, (exec_str, setup_str) in dict_exec_setup.items():
                timer = Timer(exec_str, setup_str)
                period = timer.timeit(number=number)

                exp_results[key][n][str(density)] = period
                pbar.set_description(f"{key}: {n} x {n}; {density} done")
                pbar.update(1)

            for key, (exec_str, setup_str) in dict_exec_setup.items():
                if key != 'torch_matmul':
                    result[key][n][str(density)] = (exp_results['torch_matmul'][n][str(density)] /
                                                    exp_results[key][n][str(density)] - 1) * 100
    pbar.close()
    return exp_results, result


if __name__ == "__main__":
    for use_cuda in [True]:
        tag = f'{torch.cuda.get_device_name()}' if use_cuda and torch.cuda.is_available() else 'cpu'
        print('=' * 8 + f'{tag}' + '=' * 8)

        n_itrs = 100
        densities = [1, 0.5, 0.1, 0.05, 0.025, 0.015, 0.01, 0.005, 0.001, 1e-4, 1e-5]
        exp_results, result = _test_sparse_matrix(n_itrs, densities, use_cuda)

        dfs = {key: pd.DataFrame.from_dict(dic) for key, dic in exp_results.items()}
        dfs_comp = {key: pd.DataFrame.from_dict(dic) for key, dic in result.items()}
        for key, df in dfs.items():
            print(f'======== {key}')
            print(df.to_string())

            df.to_csv(f"{key}_{tag}_n{n_itrs}.csv")
            if key != 'torch_matmul':
                dfs_comp[key].to_csv(f'Comp_{key}_vs_torch_matmul_{tag}_n{n_itrs}.csv')
                print(f"." * 16 + " faster than torch_matmul by")
                print(dfs_comp[key].to_string())
