import os
import datetime
import functions

# Set parameters
nc_files = os.listdir("/home/lucia/projects/FORMES/rainfall-pde-ml/data/ERA5/all/")
nc_files.sort()
for i in range(len(nc_files)):
    nc_files[i] = "/home/lucia/projects/FORMES/rainfall-pde-ml/data/ERA5/all/" + nc_files[i]

out_path = "/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_npy/"
daily_out_path = "/home/lucia/projects/FORMES/rainfall-pde-ml/data/era5_daily_npy/"
first = datetime.datetime(1900, 1, 1, hour=0, minute=0, second=0)  # time units for ERA5 data are hours since this date
mask = False

# Get dataset information
functions.print_info(nc_files)
#exit(0)
#variable_names = {"u10": "wind_u", "v10": "wind_v", "d2m": "dewTemp", "t2m": "temperature", "e": "evaporation",
#                  "z": "geopotential", "sp": "pressure", "tcc": "totalCloudClover", "tp": "precipitation",
#                  "fdir": "solarRadiation"}

variable_names = {"t2m": "temperature", "sp": "pressure", "tp": "precipitation", "d2m": "dewTemp"}
laplace_vars = ["t2m", "sp", "tp", "d2m"]

#for vname in variable_names.keys():
#    functions.count_masked_variable(nc_files, vname)

time = functions.get_time(first, nc_files)

# Load weather variables into a dictionary
created_wind = False
saved_wind = False

for vname in variable_names.keys():
    print("Processing variable", vname, ":", variable_names[vname])
    # One variable at a time, to conserve memory
    precip_series = dict()

    if (vname == "u10" or vname == "v10") and not created_wind:
        functions.create_wind_series(nc_files, precip_series, use_mask=mask, start_from=29220)
        created_wind = True

    elif (vname == "u10" or vname == "v10") and created_wind:
        pass
    else:
        functions.create_series(nc_files, vname, precip_series, use_mask=mask, start_from=29220)

    # Rescale rainfall from m to cm
    if vname == "tp":
        functions.scale_variable(precip_series, "tp", 100)

    laplace = functions.compute_laplacian(precip_series[vname], vname, (10, 12))  # testing
    print(laplace.shape)
    print(laplace)
    exit(0)

    # Resize variables
    resized_series = functions.resize_variables(precip_series)

    # Create new directory for each variable & save to daily files
    if (vname == "u10" or vname == "v10") and not saved_wind:  # wind only
        # Original size
        newdir = daily_out_path + "wind/"
        os.mkdir(newdir)
        functions.save_to_file(newdir, time, precip_series, "wind", one_series=False, resized=False, use_mask=mask)

        # Resized
        newdir = daily_out_path + "wind_resized/"
        os.mkdir(newdir)
        functions.save_to_file(newdir, time, resized_series, "wind", one_series=False, resized=False, use_mask=mask)

        saved_wind = True
    elif (vname == "u10" or vname == "v10") and saved_wind:
        pass
    else:
        # Original size
        newdir = daily_out_path + variable_names[vname] + "/"
        os.mkdir(newdir)
        functions.save_to_file(newdir, time, precip_series, vname, one_series=False, resized=False, use_mask=mask)

        # Resized
        newdir = daily_out_path + variable_names[vname] + "_resized/"
        os.mkdir(newdir)
        functions.save_to_file(newdir, time, resized_series, vname, one_series=False, resized=False, use_mask=mask)
