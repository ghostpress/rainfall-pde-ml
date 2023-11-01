import os
import datetime
import netCDF4 as nc
import numpy as np


def get_files(topdir):
    flist = []
    files = os.listdir(topdir)
    files.sort()

    for f in files:
        flist.append(topdir + f)

    return flist


def print_info(files):
    nc_obj = nc.Dataset(files[0])
    print("Creating data object from .nc file:")
    print(nc_obj)
    print(nc_obj.variables)
    pass


def create_wind_series(files, variable_series, use_mask=False):
    series_u = nc.Dataset(files[0]).variables["u10"][:]
    series_v = nc.Dataset(files[0]).variables["v10"][:]

    series = np.ma.masked_array([series_u, series_v])

    if not use_mask and type(series) == np.ma.masked_array:
        series = unmask(series)

    series.resize((series_u.shape[0], 2, series_u.shape[1], series_u.shape[2]))

    for i in range(len(files)):
        if i == 0:
            continue

        nc_obj = nc.Dataset(files[i])

        var_u = nc_obj.variables["u10"][:]
        var_v = nc_obj.variables["v10"][:]

        if len(var_u.shape) > 3:
            var_u = var_u[:,0,:,:]

        if len(var_v.shape) > 3:
            var_v = var_v[:,0,:,:]

        var = np.ma.masked_array([var_u, var_v])

        if not use_mask and type(var) == np.ma.masked_array:
            var = unmask(var)

        var.resize((var_u.shape[0], 2, var_u.shape[1], var_u.shape[2]))
        series = np.vstack((series, var))

    variable_series["wind"] = series
    pass


def create_series(files, variable, variable_series, use_mask=False):

    if variable == "wind":
        create_wind_series()
    else:
        series = nc.Dataset(files[0]).variables[variable][:]

        if not use_mask and type(series) == np.ma.masked_array:
            series = unmask(series)

        for i in range(len(files)):
            if i == 0:
                continue

            nc_obj = nc.Dataset(files[i])
            var = nc_obj.variables[variable][:]

            if not use_mask and type(var) == np.ma.masked_array:
                var = unmask(var)

            # Keep just the ERA5 version, which is index 0 in the 2nd dimension
            if len(var.shape) > 3:
                var = var[:,0,:,:]

            series = np.vstack((series, var))

        variable_series[variable] = series
    pass


def count_masked_variable(files, variable):
    for f in files:
        nc_obj = nc.Dataset(f)
        var = nc_obj[variable][:]

        # Keep just the ERA5 version, which is index 0 in the 2nd dimension
        if len(var.shape) > 3:
            var = var[:, 0, :, :]

        masked = np.ma.masked_array.count(var)

        print("{num} masked values in file {file} along {v}".format(num=masked, file=f, v=variable))
    pass


def get_time(first_date, files):
    time_vec = []

    for f in files:
        nc_obj = nc.Dataset(f)
        time = nc_obj.variables["time"][:]

        for t in time:
            time_vec.append(datetime.timedelta(hours=int(t)) + first_date)

    return time_vec


def unmask(arr, use_mean=True, new_val=None):
    if arr.mask.sum() == 0:
        return arr.data
    else:
        mask = arr.mask

        if use_mean:
            mean = np.mean(arr)
            arr[mask] = mean
        else:
            arr[mask] = new_val

        return arr.data


def scale_variable(variable_series, variable, factor):
    scaled = variable_series[variable] * factor
    variable_series[variable] = scaled
    return variable_series


def add_to_variable(variable_series, variable, summand):
    summed = variable_series[variable] + summand
    variable_series[variable] = summed
    return variable_series


def pad_variables(variable_series):
    print("Padding variables.")
    padded = variable_series.copy()

    for name, var in padded.items():

        # TODO: generalize this for initial shapes other than [:, 30, 23]
        if len(var.shape) == 3:  # non-wind variables
            new_shape = (var.shape[0], 64, 64)
            pad = np.zeros(new_shape)
            pad[:, 16:46, 20:43] = var
        else:
            new_shape = (var.shape[0], 2, 64, 64)
            pad = np.zeros(new_shape)
            pad[:, :, 16:46, 20:43] = var

        padded[name] = pad
        print(name, ", new shape: ", pad.shape)
    return padded


def save_to_file(outdir, time, variable_series, variable, one_series=True, padded=False, use_mask=False):
    print("Saving data to files in", outdir)

    if one_series:
        print("Saving one series.")
        if variable in variable_series.keys():
            date_min = time[0].strftime("%Y-%m-%d")
            date_max = time[-1].strftime("%Y-%m-%d")
            fname = variable + "_" + "6h" + "_" + date_min + "_" + date_max

            if padded:
                fname += "_padded"

            if use_mask:
                fname += "_masked.npy"
            else:
                fname += "_filled.npy"

            np.save(outdir + fname, variable_series[variable])

        else:
            create_series(variable, variable_series, use_mask)
            save_to_file(outdir, time, variable_series, variable, one_series, padded, use_mask)
    else:
        print("Saving individual daily files.")
        if variable in variable_series.keys():
            date_min = time[0]
            var_series = variable_series[variable]
            current_date = date_min
            i = 0

            while i < len(var_series):
                day_arr = var_series[i:i+4, :, :]
                fname = variable + "_fullDay_" + current_date.strftime("%Y%m%d")

                if use_mask:
                    fname += "_masked.npy"
                else:
                    fname += "_filled.npy"

                np.save(outdir + fname, day_arr)
                i += 4
                current_date = current_date + datetime.timedelta(days=1)

        else:
            create_series(variable, variable_series, use_mask)
            save_to_file(outdir, time, variable_series, variable, one_series, padded, use_mask)

    print("Done.")
