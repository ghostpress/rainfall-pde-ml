import numpy as np
from src.datasets.WeatherDataset import WeatherDataset


class NemoDataset(WeatherDataset):

    def __init__(self, naming_conv, variable_ids, variable_files, history):
        super().__init__(naming_conv, variable_ids, variable_files, history)

    def _search_by_region(self, files, region):
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

    def _all_regions(self, files):
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

    def get_pairs(self, variable, use_wind=False):#(self, files, history=1, use_wind=False):
        """Method to separate data into inputs (X) and ends (y), for example to use 4 previous days (hist=4) to
        predict the next day. The "pairs" are pairs of (X,y) inputs and ends. This method works on one region at a time.

        Parameters
        ----------
        files : list : list of files from which to get pairs
        history : int : number of previous observations to predict future observation

        Returns
        -------
        inps : np.ndarray : filenames for inputs
        ends : list : filenames for ends
        """
        files = self.variable_files[variable]
        assert(self.file_naming_convention == "NEMO_npy")

        regions = self._all_regions(files)
        final_inps = []
        final_ends = []

        for reg in regions:
            region_files = self._search_by_region(files, reg)

            reg_n = len(region_files)
            reg_inps = []
            reg_ends = region_files[self.history:]

            for i in range(reg_n - self.history):
                reg_inps.append(region_files[i:i + self.history])

            final_inps.append(reg_inps)
            final_ends.append(reg_ends)

        final_inps = np.array(final_inps)
        final_ends = np.array(final_ends)

        # Flatten arrays along region axis, while still keeping regions in correct order for (X,y) pairs
        #print("inps", final_inps.shape)
        #print("ends", final_ends.shape)
        nreg = final_inps.shape[0]
        ndays = final_inps.shape[1]
        inps = final_inps.reshape((nreg*ndays, final_inps.shape[2]))
        ends = final_ends.flatten()

        return inps, ends

    def train_val_test_split(self, split):
        assert len(split) == 3, "Please include a % split for train, validation, and test sets."

        train_files = dict()
        val_files = dict()
        test_files = dict()

        for i in range(len(self.variable_ids)):
            print(f"Splitting dataset for variable {i}: {self.variable_ids[i]}")
            vname = self.variable_ids[i]
            cutoffs = self._train_val_test_cutoffs(split, vname)  # FIXME: error here

            all_files = self.variable_files[vname]
            all_files.sort()

            train, val, test = [], [], []

            for f in all_files:
                file_date = self._get_date_from_filename(f)
                if file_date <= cutoffs[0]:
                    train.append(f)
                elif (file_date > cutoffs[0]) & (file_date <= cutoffs[1]):
                    val.append(f)
                else:
                    test.append(f)

            train_files[vname] = train
            val_files[vname] = val
            test_files[vname] = test

        TrainingDataset = NemoDataset(naming_conv=self.file_naming_convention, variable_files=train_files,
                                      variable_ids=self.variable_ids, history=self.history)
        ValidationDataset = NemoDataset(naming_conv=self.file_naming_convention, variable_files=val_files,
                                        variable_ids=self.variable_ids, history=self.history)
        TestingDataset = NemoDataset(naming_conv=self.file_naming_convention, variable_files=test_files,
                                     variable_ids=self.variable_ids, history=self.history)

        return TrainingDataset, ValidationDataset, TestingDataset
