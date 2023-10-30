import datetime
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from src.dataloader.ERA5Loader import ERA5Loader
from src.dataloader.ERA5Data import ERA5Data

path = "/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_npy/"  # path to ERA5 series numpy files - should contain one .npy file per variable of interest for all data
daily_outdir = "/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_daily_npy/"  # path to ERA5 daily numpy files for the variable of interest - should contain one .npy file per day
start = datetime.datetime(1900, 1, 1, hour=0, minute=0, second=0)  # time units for ERA5 data are hours since this date

Data = ERA5Data("/home/lucia/projects/FORMES/rainfall-pde-ml/data/ERA5/", first_date=start, use_mask=False)
Data.print_info()
Data.create_series("wind")
#Data.create_series("u10")
#Data.create_series("v10")
#Data.create_series("tp")
#Data.create_series("t2m")

Data.pad_variables()

#Data.save_to_file(path, "t2m")
#Data.save_to_file(path, "tp")
Data.save_to_file(path, "wind")

#ERA5DL = ERA5Loader(daily_outdir + "precipitation/", "ERA5_npy", [0.1, 0.1, 0.1], 32, hist=4)
#precip_training = ERA5DL.create_training()
#img1 = np.load("/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_npy/t2m_6h_2000-01-01_2023-09-07_padded_filled.npy")
#img2 = np.load("/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_npy/tp_6h_2000-01-01_2023-09-07_padded_filled.npy")
#imgw = np.load("/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_npy/wind_6h_2000-01-01_2023-09-07_padded_filled.npy")

#plt.imshow(img1[0])
#plt.show()
#plt.imshow(img2[0])
#plt.show()
#print(img.shape)
#plt.imshow(imgw[0][0])  # u
#plt.show()
#plt.imshow(imgw[0][1])  # v
#plt.show()

print("Done.")
