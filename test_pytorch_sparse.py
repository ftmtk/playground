import torch
import numpy as np
import pandas as pd
from timeit import Timer
from collections import defaultdict


def _get_adjacent_matrix_rand(density: float = 0.1, nrows: int = 2048, ncols: int = 2048, dtype: torch.dtype = torch.int32):
    nedges = int(np.ceil(density * nrows * ncols))
    # Initialize a new adjacent matrix
    adjmat = np.zeros(nrows * ncols)
    locs = np.random.choice(len(adjmat), size=nedges, replace=False)
    adjmat[locs] = 1

    return torch.as_tensor(adjmat.reshape(nrows, ncols), dtype=dtype)


def _test_sparse_matrix(number: int = 10, use_cuda: bool = False):
    def _torch_matmul(dense_A, dense_B):
        return torch.matmul(dense_A, dense_B)

    def _tensor_matmul(dense_A, dense_B):
        return dense_A.matmul(dense_B)

    def _torch_einsum(dense_A, dense_B):
        return torch.einsum("ik,kj->ij", dense_A, dense_B)

    def _torch_sparse_mm(sparse_A, dense_B):
        return torch.sparse.mm(sparse_A, dense_B)

    dict_funcs = {'torch_matmul': _torch_matmul,
                  'torch_einsum': _torch_einsum,
                  'tensor_matmul': _tensor_matmul,
                  'torch_sparse_mm': _torch_sparse_mm}

    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    exp_results = {key: defaultdict(dict) for key in dict_funcs.keys()}
    result = {key: defaultdict(dict) for key in dict_funcs.keys()}

    from_dim, to_dim = 8, 14
    decimals = 5
    for p in range(from_dim, to_dim):
        n = 2**p
        dense_matrix = torch.rand(size=[n, n], dtype=torch.float32).to(device)
        for decimal in range(decimals):
            density = 10**(-decimal)
            dense_adjmat = _get_adjacent_matrix_rand(density, n, n, torch.float32).to(device)
            sparse_adjmat = dense_adjmat.to_sparse().to(device)

            for key, func in dict_funcs.items():
                timer = None
                if 'sparse' not in key:
                    timer = Timer(lambda: func(dense_adjmat, dense_matrix))
                else:
                    timer = Timer(lambda: func(sparse_adjmat, dense_matrix))
                period = timer.timeit(number=number)

                exp_results[key][n][decimal] = period
            for key in ['torch_einsum', 'tensor_matmul', 'torch_sparse_mm']:
                result[key][n][decimal] = exp_results['torch_matmul'][n][decimal] / exp_results[key][n][decimal]

    return exp_results, result


if __name__ == "__main__":
    for use_cuda in [True]:
        tag = f'{torch.cuda.get_device_name()}' if use_cuda and torch.cuda.is_available() else 'cpu'
        print('=' * 8 + f'{tag}' + '=' * 8)

        exp_results, result = _test_sparse_matrix(20, use_cuda)

        dfs = {key: pd.DataFrame.from_dict(dic) for key, dic in exp_results.items()}
        dfs_comp = {key: pd.DataFrame.from_dict(dic) for key, dic in result.items()}
        for key, df in dfs.items():
            print(f'======== {key}')
            print(df.to_string())

            df.to_csv(f"{key}_{tag}.csv")
            if key != 'torch_matmul':
                dfs_comp[key].to_csv(f'Comp_{key}_vs_torch_matmul_{tag}.csv')
                print(f"=" * 16 + " compare with torch_matmul")
                print(dfs_comp[key].to_string())
