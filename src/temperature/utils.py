import torch
import numpy as np


def create_example_loader(X_files, y_files, batchsize=1, dtype=torch.FloatTensor):
    X = []
    y = []

    for region in X_files:
        region_X = []

        for X_file in region:
            x = np.load(X_file)
            region_X.append(x)

        X.append(region_X)

    for region in y_files:
        y.append(np.load(region))

    X = torch.from_numpy(np.array(X)).type(dtype)
    y = torch.from_numpy(np.array(y)).type(dtype)

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=batchsize)

    return loader


def check_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    return device