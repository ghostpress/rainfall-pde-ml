import os
import datetime
import functions

# Set parameters
nc_files = ["/home/lucia/projects/FORMES/rainfall-pde-ml/data/ERA5/file1.nc",
            "/home/lucia/projects/FORMES/rainfall-pde-ml/data/ERA5/file2.nc"]
out_path = "/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_npy/"
daily_out_path = "/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_daily_npy/"
first = datetime.datetime(1900, 1, 1, hour=0, minute=0, second=0)  # time units for ERA5 data are hours since this date
mask = False

# Get dataset information
functions.print_info(nc_files)
functions.count_masked_variable(nc_files, "t2m")
time = functions.get_time(first, nc_files)

# Load weather variables into a dictionary
precip_series = dict()
functions.create_series(nc_files, "t2m", precip_series, use_mask=mask)  # temperature at 2m
functions.create_wind_series(nc_files, precip_series, use_mask=mask)    # wind, u and v components
functions.create_series(nc_files, "tp", precip_series, use_mask=mask)   # total precipitation
functions.create_series(nc_files, "sp", precip_series, use_mask=mask)   # surface pressure

# Rescale rainfall from m to cm
functions.scale_variable(precip_series, "tp", 100)

resized_series = functions.resize_variables(precip_series)

# Save originally-sized arrays to daily files (in new directory for each variable)
functions.save_to_file(daily_out_path + "wind/", time, precip_series, "wind", one_series=False, resized=False, use_mask=mask)
functions.save_to_file(daily_out_path + "pressure/", time, precip_series, "sp", one_series=False, resized=False, use_mask=mask)

# Save resized arrays to daily files (in new directory for each variable)
functions.save_to_file(daily_out_path + "temperature_resized/", time, resized_series, "t2m", one_series=False, resized=True, use_mask=mask)
functions.save_to_file(daily_out_path + "wind_resized/", time, resized_series, "wind", one_series=False, resized=True, use_mask=mask)
functions.save_to_file(daily_out_path + "precipitation_resized/", time, resized_series, "tp", one_series=False, resized=True, use_mask=mask)
functions.save_to_file(daily_out_path + "pressure_resized/", time, resized_series, "sp", one_series=False, resized=True, use_mask=mask)

# Save full padded arrays to file
functions.save_to_file(out_path, time, resized_series, "t2m", one_series=True, resized=True, use_mask=mask)
functions.save_to_file(out_path, time, resized_series, "wind", one_series=True, resized=True, use_mask=mask)
functions.save_to_file(out_path, time, resized_series, "tp", one_series=True, resized=True, use_mask=mask)
functions.save_to_file(out_path, time, resized_series, "sp", one_series=True, resized=True, use_mask=mask)
