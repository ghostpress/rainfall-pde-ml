import datetime
import matplotlib.pyplot as plt
import numpy as np
from src.dataloader import WeatherDataLoader


class SSTLoader(WeatherDataLoader.WeatherDataLoader):

    def get_date_from_filename(self, fname):
        """Helper method to take a filename containing a date string and return the date as a datetime.datetime object.
        Implemented by each subclass because the file naming convention used for each dataset is different.

        Naming convention for SST data: sst_geo_yyyymmdd.nc_region_XX.npy

        Parameters
        ----------
        fname : str : filename

        Returns
        -------
        date : datetime.datetime : date represented by the string in the filename
        """
        assert(self.file_naming_convention == "NEMO_npy")

        date = datetime.datetime.strptime(fname[8:16], "%Y%m%d").date()
        return date

    def search_by_region(self, files, region):
        """Helper method to search a list of files for only those corresponding to a desired region.
        File naming convention: sst_geo_yyyymmdd.nc_region_XX.npy

        Parameters
        ----------
        files : list : list of filenames in which to search
        region : int : desired region

        Returns
        -------
        region_files : list : sorted list of matching files
        """
        assert (self.file_naming_convention == "NEMO_npy")

        region_files = []
        for fname in files:
            if "region_" + str(region) + ".npy" in fname:
                region_files.append(fname)

        region_files.sort()
        return region_files

    def all_regions(self, files):
        """Helper method to take a list of filenames and return a list of the unique region numbers.
        File naming convention: sst_geo_yyyymmdd.nc_region_XX.npy

        Parameters
        ----------
        files : list : data file names

        Returns
        -------
        regions : list : list of unique region numbers in the directory
        """
        assert (self.file_naming_convention == "NEMO_npy")

        files.sort()
        regions = []

        for f in files:
            start_ind = f.find("region_") + len("region_")
            end_ind = f.find(".npy")

            reg = f[start_ind:end_ind]

            if int(reg) not in regions:
                regions.append(int(reg))

        regions.sort()
        return regions

    def __get_pairs(self, files):
        """Method to separate data into inputs (X) and ends (y), for example to use 4 previous days (hist=4) to
        predict the next day. The "pairs" are pairs of (X,y) inputs and ends. This method works on one region at a time.

        Parameters
        ----------
        files : list : list of files from which to get pairs

        Returns
        -------
        inps : np.ndarray : filenames for inputs
        ends : list : filenames for ends
        """

        print("here, in SST class")
        regions = self.all_regions(files)
        final_inps = []
        final_ends = []

        for reg in regions:
            region_files = self.search_by_region(files, reg)

            reg_n = len(region_files)
            reg_inps = []
            reg_ends = region_files[self.history:]

            for i in range(reg_n - self.history):
                reg_inps.append(region_files[i:i + self.history])

            final_inps.append(reg_inps)
            final_ends.append(reg_ends)

        return np.array(final_inps), np.array(final_ends)

    def plot_example_image(self, arr):
        """Helper method to take a multidimensional array or Tensor from a dataloader and plot the image embedded
        within. Implemented by each subclass because the dimensions of the data differ.

        Parameters
        ----------
        arr : torch.Tensor : array containing image to plot
        """
        print("Example image:")
        plt.imshow(arr[0][0])
        plt.show()
