import datetime
import os
import netCDF4 as nc
import numpy as np


class ERA5Data:

    def __init__(self, topdir, first_date, use_mask):
        """Method to initialize the ERA5Data object.
        Parameters
        ----------
        topdir : str : path to parent directory holding .nc files
        first_date : datetime.datetime : the first date of the data, from the time units
        """
        self.topdir = topdir
        self.first_date = first_date
        self.use_mask = use_mask

        self.files = self.get_files()
        self.print_info()

        self.time = self.get_time()
        self.variable_series = dict()

        self.padded = False
        pass

    def get_files(self):
        flist = []
        files = os.listdir(self.topdir)
        files.sort()

        for f in files:
            flist.append(self.topdir + f)

        return flist

    def print_info(self):
        nc_obj = nc.Dataset(self.files[0])
        print("Creating data object from .nc file:")
        print(nc_obj)
        print(nc_obj.variables)
        pass

    def create_wind_series(self):
        # TODO: check series_u, v, for masked values
        series_u = nc.Dataset(self.files[0]).variables["u10"][:]
        series_v = nc.Dataset(self.files[0]).variables["v10"][:]

        series = np.ma.masked_array([series_u, series_v])

        if not self.use_mask and type(series) == np.ma.masked_array:
            series = self.unmask(series)

        series.resize((series_u.shape[0], 2, series_u.shape[1], series_u.shape[2]))

        for i in range(len(self.files)):
            if i == 0:
                continue

            nc_obj = nc.Dataset(self.files[i])

            var_u = nc_obj.variables["u10"][:]
            var_v = nc_obj.variables["v10"][:]

            if len(var_u.shape) > 3:
                var_u = var_u[:,0,:,:]

            if len(var_v.shape) > 3:
                var_v = var_v[:,0,:,:]

            var = np.ma.masked_array([var_u, var_v])

            if not self.use_mask and type(var) == np.ma.masked_array:
                var = self.unmask(var)

            var.resize((var_u.shape[0], 2, var_u.shape[1], var_u.shape[2]))
            series = np.vstack((series, var))

        self.variable_series["wind"] = series

    def create_series(self, variable):

        if variable == "wind":
            self.create_wind_series()
        else:
            series = nc.Dataset(self.files[0]).variables[variable][:]

            if not self.use_mask and type(series) == np.ma.masked_array:
                series = self.unmask(series)

            for i in range(len(self.files)):
                if i == 0:
                    continue

                nc_obj = nc.Dataset(self.files[i])
                var = nc_obj.variables[variable][:]

                if not self.use_mask and type(var) == np.ma.masked_array:
                    var = self.unmask(var)

                # Keep just the ERA5 version, which is index 0 in the 2nd dimension
                if len(var.shape) > 3:
                    var = var[:,0,:,:]

                series = np.vstack((series, var))

            self.variable_series[variable] = series

    def count_masked_variable(self, variable):
        for f in self.files:
            nc_obj = nc.Dataset(f)
            var = nc_obj[variable][:]

            # Keep just the ERA5 version, which is index 0 in the 2nd dimension
            if len(var.shape) > 3:
                var = var[:, 0, :, :]

            masked = np.ma.masked_array.count(var)

            print("{num} masked values in file {file} along {v}".format(num=masked, file=f, v=variable))
        pass

    def get_time(self):
        time_vec = []

        for f in self.files:
            nc_obj = nc.Dataset(f)
            time = nc_obj.variables["time"][:]

            for t in time:
                time_vec.append(datetime.timedelta(hours=int(t)) + self.first_date)

        return time_vec

    @staticmethod
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

    def scale_variable(self, variable, factor):
        if variable in self.variable_series.keys():
            scaled = self.variable_series[variable] * factor
            self.variable_series[variable] = scaled
        else:
            self.create_series(variable)
            self.scale_variable(variable, factor)

        pass

    def add_to_variable(self, variable, summand):
        if variable in self.variable_series.keys():
            scaled = self.variable_series[variable] + summand
            self.variable_series[variable] = scaled
        else:
            self.create_series(variable)
            self.scale_variable(variable, summand)

        pass

    def save_to_file(self, outdir, variable, one_series=True):
        print("Saving data to files in", outdir)

        if one_series:
            print("Saving one series.")
            if variable in self.variable_series.keys():
                date_min = self.time[0].strftime("%Y-%m-%d")
                date_max = self.time[-1].strftime("%Y-%m-%d")
                fname = variable + "_" + "6h" + "_" + date_min + "_" + date_max

                if self.padded:
                    fname += "_padded"

                if self.use_mask:
                    fname += "_masked.npy"
                else:
                    fname += "_filled.npy"

                np.save(outdir + fname, self.variable_series[variable])

            else:
                self.create_series(variable)
                self.save_to_file(outdir, variable, one_series)
        else:
            print("Saving individual daily files.")
            if variable in self.variable_series.keys():
                date_min = self.time[0]
                var_series = self.variable_series[variable]
                current_date = date_min
                i = 0

                while i < len(var_series):
                    day_arr = var_series[i:i+4, :, :]
                    fname = variable + "_fullDay_" + current_date.strftime("%Y%m%d")

                    if self.use_mask:
                        fname += "_masked.npy"
                    else:
                        fname += "_filled.npy"

                    np.save(outdir + fname, day_arr)
                    i += 4
                    current_date = current_date + datetime.timedelta(days=1)

            else:
                self.create_series(variable)
                self.save_to_file(outdir, variable, one_series)

        pass

    def pad_variables(self):
        print("Padding variables.")

        for name, var in self.variable_series.items():

            # TODO: generalize this for initial shapes other than [:, 30, 23]
            if len(var.shape) == 3:  # non-wind variables
                new_shape = (var.shape[0], 64, 64)
                pad = np.zeros(new_shape)
                pad[:, 16:46, 20:43] = var
            else:
                new_shape = (var.shape[0], 2, 64, 64)
                pad = np.zeros(new_shape)
                pad[:, :, 16:46, 20:43] = var

            self.variable_series[name] = pad
            print(name, ", new shape: ", pad.shape)

        self.padded = True
        pass
