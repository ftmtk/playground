import torch
import numpy as np
import pandas as pd
from timeit import Timer
from collections import defaultdict

from torch.cuda import is_available


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

    def _torch_sparse_mm(sparse_A, dense_B):
        return torch.sparse.mm(sparse_A, dense_B)

    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    exp_results = {'dense_multiply': defaultdict(dict), 'sparse_multiply': defaultdict(dict)}
    result = defaultdict(dict)

    from_dim, to_dim = 8, 14
    decimals = 5
    for p in range(from_dim, to_dim):
        n = 2**p
        dense_matrix = torch.rand(size=[n, n], dtype=torch.float32).to(device)
        for decimal in range(decimals):
            density = 10**(-decimal)
            dense_adjmat = _get_adjacent_matrix_rand(density, n, n, torch.float32).to(device)
            sparse_adjmat = dense_adjmat.to_sparse().to(device)

            timer_matmul = Timer(lambda: _torch_matmul(dense_adjmat, dense_matrix))
            timer_sparse_mm = Timer(lambda: _torch_sparse_mm(sparse_adjmat, dense_matrix))
            t_matmul = timer_matmul.timeit(number=number)
            t_sparse_mm = timer_sparse_mm.timeit(number=number)

            exp_results['dense_multiply'][n][decimal] = t_matmul
            exp_results['sparse_multiply'][n][decimal] = t_sparse_mm
            result[n][decimal] = t_matmul / t_sparse_mm

    df_matmul = pd.DataFrame.from_dict(exp_results['dense_multiply'])
    df_sparse_mm = pd.DataFrame.from_dict(exp_results['sparse_multiply'])
    df_dense_vs_sparse = pd.DataFrame.from_dict(result)
    return df_matmul, df_sparse_mm, df_dense_vs_sparse


if __name__ == "__main__":
    for use_cuda in [True, False]:
        tag = f'{torch.cuda.get_device_name()}' if use_cuda and torch.cuda.is_available() else 'cpu'
        print('=' * 8 + f'{tag}' + '=' * 8)

        df_matmul, df_sparse_mm, df_dense_vs_sparse = _test_sparse_matrix(20, use_cuda)
        print(df_matmul.to_string())
        print(df_sparse_mm.to_string())
        print(df_dense_vs_sparse.to_string())

        df_matmul.to_csv(f'matmul_{tag}.csv')
        df_sparse_mm.to_csv(f'sparse_mm_{tag}.csv')
        df_dense_vs_sparse.to_csv(f'dense_vs_sparse_{tag}.csv')
