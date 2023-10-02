import os
import netCDF4 as nc
import numpy as np
import datetime


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

        # Note: assumes that the geographic area of the data will remain the same throughout the project
        self.latitude = nc.Dataset(self.files[0]).variables["latitude"][:].data
        self.longitude = nc.Dataset(self.files[0]).variables["longitude"][:].data

        self.time = self.get_time()
        self.variable_series = dict()
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

    def create_series(self, variable):

        series = nc.Dataset(self.files[0]).variables[variable][:]
        if not self.use_mask:
            series = ERA5Data.unmask(series)

        for i in range(len(self.files)):
            if i == 0:
                continue

            nc_obj = nc.Dataset(self.files[i])
            var = nc_obj.variables[variable][:]

            if not self.use_mask:
                var = ERA5Data.unmask(var)

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

    def save_to_file(self, outdir, variable):
        print("Saving data to files in", outdir)

        if variable in self.variable_series.keys():
            date_min = self.time[0].strftime("%Y-%m-%d")
            date_max = self.time[-1].strftime("%Y-%m-%d")
            fname = variable + "_" + "6h" + "_" + date_min + "_" + date_max

            if self.use_mask:
                fname += "_masked.npy"
            else:
                fname += "_filled.npy"

            np.save(outdir + fname, self.variable_series[variable])

        else:
            self.create_series(variable)
            self.save_to_file(outdir, variable)

        pass
