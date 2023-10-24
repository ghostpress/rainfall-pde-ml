import torch
import numpy as np


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


def datadir(x):
    return "/projectnb/labci/Lucia/data/" + x


def create_example_loader(batchsize=1, dtype=torch.FloatTensor):
    X = []
    y = []

    reg3_X = [datadir("sst_npy/sst_geo_20210605.nc_region_3.npy"),
              datadir("sst_npy/sst_geo_20210606.nc_region_3.npy"),
              datadir("sst_npy/sst_geo_20210607.nc_region_3.npy"),
              datadir("sst_npy/sst_geo_20210608.nc_region_3.npy")]

    reg22_X = [datadir("sst_npy/sst_geo_20210605.nc_region_22.npy"),
               datadir("sst_npy/sst_geo_20210606.nc_region_22.npy"),
               datadir("sst_npy/sst_geo_20210607.nc_region_22.npy"),
               datadir("sst_npy/sst_geo_20210608.nc_region_22.npy")]

    reg29_X = [datadir("sst_npy/sst_geo_20210605.nc_region_29.npy"),
               datadir("sst_npy/sst_geo_20210606.nc_region_29.npy"),
               datadir("sst_npy/sst_geo_20210607.nc_region_29.npy"),
               datadir("sst_npy/sst_geo_20210608.nc_region_29.npy")]

    reg44_X = [datadir("sst_npy/sst_geo_20210605.nc_region_44.npy"),
               datadir("sst_npy/sst_geo_20210606.nc_region_44.npy"),
               datadir("sst_npy/sst_geo_20210607.nc_region_44.npy"),
               datadir("sst_npy/sst_geo_20210608.nc_region_44.npy")]

    examples_X = [reg3_X, reg22_X, reg29_X, reg44_X]
    examples_y = [datadir("sst_npy/sst_geo_20210609.nc_region_3.npy"),
                  datadir("sst_npy/sst_geo_20210609.nc_region_22.npy"),
                  datadir("sst_npy/sst_geo_20210609.nc_region_29.npy"),
                  datadir("sst_npy/sst_geo_20210609.nc_region_44.npy")]

    for region in examples_X:
        region_X = []

        for X_file in region:
            x = np.load(X_file)
            region_X.append(x)

        X.append(region_X)

    for region in examples_y:
        y.append(np.load(region))

    X = torch.from_numpy(np.array(X)).type(dtype)
    y = torch.from_numpy(np.array(y)).type(dtype)

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=batchsize)

    return loader

