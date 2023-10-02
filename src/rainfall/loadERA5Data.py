import datetime
import numpy as np
import netCDF4

from src.dataloader.ERA5Data import ERA5Data

path = "/home/lucia/projects/FORMES/rainfall-pde-ml/data/ERA5/"
outdir = "/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_npy/"
start = datetime.datetime(1900, 1, 1, hour=0, minute=0, second=0)

E5_filled = ERA5Data(path, start, use_mask=False)
E5_masked = ERA5Data(path, start, use_mask=True)

E5_filled.create_series("tp")
E5_filled.create_series("t2m")
E5_masked.create_series("tp")
E5_masked.create_series("t2m")

# Convert to desired units
E5_filled.scale_variable("tp", 100)  # precipitation: m -> cm
E5_filled.add_to_variable("t2m", -273.15)  # temperature: K -> degrees C
E5_masked.scale_variable("tp", 100)  # precipitation: m -> cm
E5_masked.add_to_variable("t2m", -273.15)  # temperature: K -> degrees C

# Save to file
E5_filled.save_to_file(outdir, "tp")
E5_filled.save_to_file(outdir, "t2m")
# E5_masked.save_to_file(outdir, "tp")  # MaskedArray.tofile() not implemented yet
# E5_masked.save_to_file(outdir, "t2m") # MaskedArray.tofile() not implemented yet

# Test that they were saved correctly
precip = np.load("/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_npy/t2m_6h_2000-01-01_2023-09-07_filled.npy")
temp = np.load("/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_npy/tp_6h_2000-01-01_2023-09-07_filled.npy")

print(precip.shape)
print(temp.shape)

print(np.mean(precip))
print(np.mean(temp))

print("Done.")
