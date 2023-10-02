import datetime
import numpy as np
import netCDF4
import os

from src.dataloader.ERA5Data import ERA5Data
from src.dataloader.ERA5Loader import ERA5Loader
from src.dataloader.SSTLoader import SSTLoader

path = "/home/lucia/projects/FORMES/rainfall-pde-ml/data/ERA5/"
series_outdir = "/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_npy/"
discrete_outdir = "/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_daily_npy/"
start = datetime.datetime(1900, 1, 1, hour=0, minute=0, second=0)

#E5_filled = ERA5Data(path, start, use_mask=False)

#E5_filled.create_series("tp")
#E5_filled.create_series("t2m")
#E5_filled.create_series("latitude")
#E5_filled.create_series("longitude")

# Convert to desired units
#E5_filled.scale_variable("tp", 100)  # precipitation: m -> cm
#E5_filled.add_to_variable("t2m", -273.15)  # temperature: K -> degrees C

# Save to file - series
#E5_filled.save_to_file(series_outdir, "tp", one_series=True)
#E5_filled.save_to_file(series_outdir, "t2m", one_series=True)
#E5_filled.save_to_file(series_outdir, "latitude", one_series=True)
#E5_filled.save_to_file(series_outdir, "longitude", one_series=True)

# Save to file - daily discrete
#E5_filled.save_to_file(discrete_outdir + "precipitation/", "tp", one_series=False)
#E5_filled.save_to_file(discrete_outdir + "temperature/", "t2m", one_series=False)

# Test that the series were saved correctly
#print("Testing series.")
#precip = np.load("/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_npy/t2m_6h_2000-01-01_2023-09-07_filled.npy")
#temp = np.load("/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_npy/tp_6h_2000-01-01_2023-09-07_filled.npy")
#print(precip.shape)
#print(temp.shape)

# Test that the daily files were saved correctly
#print("Testing daily.")
#files = os.listdir(discrete_outdir + "precipitation/")
#num_days = len(files)
#print(num_days)
#assert(num_days == 34604/4)
#test1 = np.load(discrete_outdir + "precipitation/" + files[0])
#print(test1.shape)

#files = os.listdir(discrete_outdir + "temperature/")
#num_days = len(files)
#print(num_days)
#assert(num_days == 34604/4)
#test2 = np.load(discrete_outdir + "temperature/" + files[0])
#print(test2.shape)

print("Testing ERA5Loader")

ERA5DL = ERA5Loader(discrete_outdir + "precipitation/", "ERA5_npy", [0.8, 0.1, 0.1], 32, hist=4, shuffle=True)
print(ERA5DL.get_date_from_filename("tp_fullDay_20000107_filled.npy"))
precip_training = ERA5DL.create_training()

print("Done.")
