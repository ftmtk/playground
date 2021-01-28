# Benchmarking PyTorch tensor multiplication

Tested on
``` 
Python 3.8.5
PyTorch 1.5.1
```

## Running script
Go to `test_pytorch_sparse.py` and change desired testing matrix sizes and densities
```
sizes = [128, 1024, 2048, 4096, 8192]
densities = [10**p for p in np.linspace(-4, 0, 10)]
```

Make sure one is running the script over a machine with GPU and CUDA available. Simple do
```
python test_pytorch_sparse.py
```

The script will generate a json file.

## Visualizing results
Go to `plot_test_pytorch_sparse.py`, change the json filename
```
filename = 'test_sparse_matrix_gpu_sizes128-1024-4096-8192.json'
```

Simply do
```
python plot_test_pytorch_sparse.py
```

The script will generate analysis plots in `.png`.