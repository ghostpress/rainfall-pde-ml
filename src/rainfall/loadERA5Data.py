import datetime

from src.dataloader.ERA5Loader import ERA5Loader

path = ""  # path to ERA5 series numpy files - should contain one .npy file per variable of interest for all data
daily_outdir = ""  # path to ERA5 daily numpy files for the variable of interest - should contain one .npy file per day
start = datetime.datetime(1900, 1, 1, hour=0, minute=0, second=0)  # time units for ERA5 data are hours since this date

ERA5DL = ERA5Loader(daily_outdir + "precipitation/", "ERA5_npy", [0.8, 0.1, 0.1], 32, hist=4, shuffle=True)
precip_training = ERA5DL.create_training()

print("Done.")
