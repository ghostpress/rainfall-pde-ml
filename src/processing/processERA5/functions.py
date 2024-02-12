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


def create_wind_series(files, variable_series, use_mask=False, start_from=0):
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

    series = series[start_from:, :, :, :]
    variable_series["wind"] = series
    pass


def create_series(files, variable, variable_series, use_mask=False, start_from=0):
    if variable == "wind":
        create_wind_series(files, variable_series, use_mask=use_mask, start_from=start_from)
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

        series = series[start_from:, :, :]
        variable_series[variable] = series
    pass


def count_masked_variable(files, variable):
    for f in files:
        nc_obj = nc.Dataset(f)
        var = nc_obj[variable][:]

        # Keep just the ERA5 version, which is index 0 in the 2nd dimension
        if len(var.shape) > 3:
            var = var[:, 0, :, :]

        masked = np.ma.count_masked(var)
        total = var.size
        print("{num} masked values in file {file} along {v}".format(num=masked, file=f, v=variable))
        print("{tot} total values in file {file} along {v}".format(tot=total, file=f, v=variable))
    pass


def get_time(first_date, files, start_from=0):
    time_vec = []

    for f in files:
        nc_obj = nc.Dataset(f)
        time = nc_obj.variables["time"][:] #[start_from:]

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


def compute_laplacian(var_array, var_name, loc):

    assert(len(loc) == 2)

    print("Computing Laplacian for variable " + var_name)
    var_shape = np.array(var_array.shape)
    row_border = var_shape[1]-1  # 29 or 63
    col_border = var_shape[2]-1  # 22 or 63

    try:
        center = var_array[:, loc[0], loc[1]]
    except IndexError:
        print("The chosen location is out of bounds, please choose indices within " + str(var_shape[1:]))
        loc = [int(x) for x in input("New location (space between integer indices): ").split()]
        center = var_array[:, loc[0], loc[1]]

    north, south, east, west = None, None, None, None

    # not on a corner/edge:
    if (loc[0] not in [0, row_border]) and (loc[1] not in [0, col_border]):
        north = var_array[:, loc[0] - 1, loc[1]]
        south = var_array[:, loc[0] + 1, loc[1]]
        east = var_array[:, loc[0], loc[1] + 1]
        west = var_array[:, loc[0], loc[1] - 1]

    # northwest corner
    elif (loc[0] == 0) and (loc[1] == 0):
        south = var_array[:, loc[0] + 1, loc[1]]
        east = var_array[:, loc[0], loc[1] + 1]

    # northeast corner
    elif (loc[0] == row_border) and (loc[1] == 0):
        south = var_array[:, loc[0] + 1, loc[1]]
        west = var_array[:, loc[0], loc[1] - 1]

    # southwest corner
    elif (loc[0] == 0) and [loc[1] == col_border]:
        north = var_array[:, loc[0] - 1, loc[1]]
        east = var_array[:, loc[0], loc[1] + 1]

    # southeast corner
    elif (loc[0] == row_border) and (loc[1] == col_border):
        north = var_array[:, loc[0] - 1, loc[1]]
        west = var_array[:, loc[0], loc[1] - 1]

    # north edge
    elif loc[0] == 0:
        south = var_array[:, loc[0] + 1, loc[1]]
        east = var_array[:, loc[0], loc[1] + 1]
        west = var_array[:, loc[0], loc[1] - 1]

    # south edge
    elif loc[0] == col_border:
        north = var_array[:, loc[0] - 1, loc[1]]
        east = var_array[:, loc[0], loc[1] + 1]
        west = var_array[:, loc[0], loc[1] - 1]

    # east edge
    elif loc[1] == row_border:
        north = var_array[:, loc[0] - 1, loc[1]]
        south = var_array[:, loc[0] + 1, loc[1]]
        west = var_array[:, loc[0], loc[1] - 1]

    # west edge
    elif loc[1] == 0:
        north = var_array[:, loc[0] - 1, loc[1]]
        south = var_array[:, loc[0] + 1, loc[1]]
        east = var_array[:, loc[0], loc[1] + 1]

    # out of bounds
    else:
        raise IndexError("The chosen location is out of bounds, please choose indices within " + var_shape[1, 2])

    laplace = center - np.mean(np.array([north, south, east, west]), axis=0)
    return laplace


def resize_variables(variable_series):
    print("Resizing variables.")
    resized = variable_series.copy()

    for name, var in resized.items():

        if len(var.shape) == 3:  # non-wind variables
            temp1 = np.repeat(var, 2, axis=1)
            temp2 = np.repeat(temp1, 3, axis=2)
            temp3 = np.pad(temp2, pad_width=((0, 0), (2, 2), (0, 0)), mode="edge")
            resized_img = temp3[:, :, 0:64]

        else:
            temp1 = np.repeat(var, 2, axis=2)
            temp2 = np.repeat(temp1, 3, axis=3)
            temp3 = np.pad(temp2, pad_width=((0, 0), (0, 0), (2, 2), (0, 0)), mode="edge")
            resized_img = temp3[:, :, :, 0:64]

        resized[name] = resized_img
        print(name, "new shape:", resized_img.shape)
    return resized


def save_to_file(outdir, time, variable_series, variable, one_series=True, resized=False, use_mask=False):
    print("Saving data to files in", outdir)

    if one_series:
        print("Saving one series.")
        if variable in variable_series.keys():
            date_min = time[0].strftime("%Y-%m-%d")
            date_max = time[-1].strftime("%Y-%m-%d")
            fname = variable + "_" + "6h" + "_" + date_min + "_" + date_max

            if resized:
                fname += "_resized"

            if use_mask:
                fname += "_masked.npy"
            else:
                fname += "_filled.npy"

            np.save(outdir + fname, variable_series[variable])

        else:
            create_series(variable, variable_series, use_mask)
            save_to_file(outdir, time, variable_series, variable, one_series, resized, use_mask)
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
            save_to_file(outdir, time, variable_series, variable, one_series, resized, use_mask)

    print("Done.")
