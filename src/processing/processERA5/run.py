import os
import datetime
import functions

# Set parameters
path_to_nc_files = ""
nc_files = os.listdir(path_to_nc_files)
out_path = ""
daily_out_path = ""
first = datetime.datetime(1900, 1, 1, hour=0, minute=0, second=0)  # time units for ERA5 data are hours since this date
mask = False

# Get dataset information
functions.print_info(nc_files)
functions.count_masked_variable(nc_files, "t2m")
time = functions.get_time(first, nc_files)

# Load rainfall, temperature, and wind variables into a dictionary
precip_series = dict()
functions.create_series(nc_files, "t2m", precip_series, use_mask=mask)
functions.create_wind_series(nc_files, precip_series, use_mask=mask)
functions.create_series(nc_files, "tp", precip_series, use_mask=mask)

# Rescale rainfall from m to cm
functions.scale_variable(precip_series, "tp", 100)

padded_series = functions.pad_variables(precip_series)

# Save padded arrays to daily files (in new directory for each variable)
functions.save_to_file(daily_out_path + "temperature_padded/", time, padded_series, "t2m", one_series=False, padded=True, use_mask=mask)
functions.save_to_file(daily_out_path + "wind_padded/", time, padded_series, "wind", one_series=False, padded=True, use_mask=mask)
functions.save_to_file(daily_out_path + "precipitation_padded/", time, padded_series, "tp", one_series=False, padded=True, use_mask=mask)

# Save full padded arrays to file
functions.save_to_file(out_path, time, padded_series, "t2m", one_series=True, padded=True, use_mask=mask)
functions.save_to_file(out_path, time, padded_series, "wind", one_series=True, padded=True, use_mask=mask)
functions.save_to_file(out_path, time, padded_series, "tp", one_series=True, padded=True, use_mask=mask)
