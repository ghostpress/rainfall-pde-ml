import datetime
import numpy as np

import src.dataloader


class SSTLoader(src.dataloader.WeatherDataLoader):

    def __init__(self):
        super().__init__()
        pass

    def get_date_from_filename(self, fname):
        assert(self.file_naming_convention == "NEMO_npy")

        date = datetime.datetime.strptime(fname[8:16], "%Y%m%d").date()
        return date

    def search_by_region(self, files, region):
        """Helper method to search a list of files for only those corresponding to
           a desired region.

           This method assumes a particular file naming convention that has been used
           for this project.

           File naming convention: sst_geo_yyyymmdd.nc_region_XX.npy

           Parameters
           ----------
           files : list : list of filenames in which to search
           region : int : desired region

           Returns
           -------
           region_files : list : sorted list of matching files
        """

        region_files = []
        for fname in files:
            if "region_" + str(region) + ".npy" in fname:
                region_files.append(fname)

        region_files.sort()

        return region_files

    def all_regions(self, files):
        """Helper method to take a list of filenames and return a list of the
        unique region numbers.

        This method assumes a particular file naming convention that has been used
        for this project.

        File naming convention: sst_geo_yyyymmdd.nc_region_XX.npy

        Parameters
        ----------
        files : list : data file names

        Returns
        -------
        regions : list : list of unique region numbers in the directory
        """

        files.sort()
        regions = []

        for f in files:
            if "region_" in f:

                start_ind = f.find("region_") + len("region_")
                end_ind = f.find(".npy")

                reg = f[start_ind:end_ind]

                if int(reg) not in regions:
                    regions.append(int(reg))

            else:
                raise ValueError("Files in this directory do not match the naming convention.")

        regions.sort()

        return regions

    def get_pairs(self, files):
        """Method to separate data into inputs (X) and ends (y), for example to
        use 4 previous days (ndays=4) to predict the next day. The "pairs" are
        pairs of (X,y) inputs and ends. This method works on one region at a time.

        Parameters
        ----------
        files : list : list of files from which to get pairs
        region : int : desired region

        Returns
        -------
        inps : np.ndarray : filenames for inputs
        ends : list : filenames for ends
        """

        regions = self.all_regions(self, files)
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
