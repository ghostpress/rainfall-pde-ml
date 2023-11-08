import os
import numpy as np
import datetime
import math
from torch.utils.data import Dataset


class WeatherDataset(Dataset):

    def __init__(self, naming_conv, variable_ids, variable_files):
        """Method to instantiate a WeatherDataset object. Takes a string describing the file naming convention for the
        data, a dictionary of variable ids and names, and a dictionary of variable names and files.

        Parameters
        ----------
        naming_conv : str : the naming convention used
        variable_ids : dict : the dictionary of variable id's : variable names
        variable_files : dict : the dictionary of variable names : variable files
        """
        assert naming_conv in ["ERA5_npy", "NEMO_npy"], f"Unsupported naming convention: {naming_conv}"
        self.file_naming_convention = naming_conv
        self.variable_ids = variable_ids
        self.variable_files = variable_files

    def __len__(self):
        """Method to return the length of the Dataset."""
        var = self.variable_ids[0]
        return len(self.variable_files[var])

    def __getitem__(self, idx):
        """Method to return the ith item in the Dataset, for each variable it contains.
        Parameters
        ----------
        idx : int : the desired index
        """

        # Case 1: dataset contains just the primary variable
        if len(self.variable_ids) == 1:
            var = self.variable_ids[0]
            return self._load_data_from_file(self.variable_files[var][idx])

        # Case 2: dataset contains the primary variable and one other variable
        elif len(self.variable_ids) == 2:
            var_0 = self.variable_ids[0]
            var_1 = self.variable_ids[1]

            item1 = self._load_data_from_file(self.variable_files[var_0][idx])
            item2 = self._load_data_from_file(self.variable_files[var_1][idx])

            return item1, item2

        # Case 3: dataset contains the primary variable and many other variables
        else:
            var_0 = self.variable_ids[0]
            items = [self._load_data_from_file(self.variable_files[var_0][idx])]

            for i in range(len(self.variable_files)):
                var_i = self.variable_ids[i+1]
                items.append(self._load_data_from_file(self.variable_files[var_i][idx]))

            return items

    def _get_date_from_filename(self, fname):
        """Helper method to take a filename containing a date string and return the date as a datetime.datetime object.

        Parameters
        ----------
        fname : str : filename

        Returns
        -------
        date : datetime.datetime : date represented by the string in the filename
        """

        if self.file_naming_convention == "ERA5_npy":
            return datetime.datetime.strptime(fname.split("_")[-2], "%Y%m%d").date()
        else:
            return datetime.datetime.strptime(fname[8:16], "%Y%m%d").date()

    def _all_days(self, files):
        """Helper method to take a list of files and return the total number of days represented by the files in
        the list.

        Parameters
        ----------
        files : list : list of file paths

        Returns
        -------
        delta : int : the number of days represented by the inputted files
        """
        files.sort()

        first = self._get_date_from_filename(files[0])
        last = self._get_date_from_filename(files[len(files) - 1])

        delta = int((last - first) / datetime.timedelta(days=1))

        return delta

    # helper method
    def _train_val_test_cutoffs(self, split, variable):
        """Helper method to create lists of filenames for the train, val, and test data splits. Uses the dates in the
        filenames to determine file order and split cutoffs.

        Parameters
        ----------
        split : list : list of fractions for each split
        variable : str : the variable to create a split dataset for

        Returns
        -------
        cutoffs : list : date cutoffs for each split
        """

        all_files = self.variable_files[variable]
        all_files.sort()
        nfiles = len(all_files)
        ndays = self._all_days(all_files)

        start_date = self._get_date_from_filename(all_files[0])
        end_date = self._get_date_from_filename(all_files[nfiles - 1])

        cutoffs = []

        for i in range(3):
            delta = math.floor(ndays * split[i])
            end = start_date + datetime.timedelta(days=delta)
            cutoffs.append(end)

            start_date = end

        # Because of rounding, some files may have been missed
        # Add these to the test split
        if cutoffs[2] < end_date:
            cutoffs[2] = end_date

        return cutoffs

    def train_val_test_split(self, split):
        """Method to split the data in a directory into training, validation, and test sets and return a new Dataset
        object. Uses the helper method train_val_test_cutoffs().
        """
        raise NotImplementedError("Please implement this method for each subclass.")

    # helper method
    def _load_data_from_file(self, filename):
        """Helper method to load a single .npy file."""
        return np.load(filename)

    def load_data_from_files(self, files):
        """Method to load data from a list of .npy files."""
        data = []
        for f in files:
            data.append(self._load_data_from_file(f))

        return np.array(data)

    def get_pairs(self, variable, history=1):
        """Method to separate data into inputs (X) and ends (y) for a given variable.
        For example: to use 4 previous days (X, hist=4) to predict the next day (y). The "pairs" are pairs of (X,y)
        inputs and ends. Implemented by each subclass because the dimensions of the data differ.

        Parameters
        ----------
        variable : str : the variable to create pairs for
        history : int : the amount of history to use for prediction

        Returns
        -------
        inps : np.ndarray : filenames for inputs
        ends : list : filenames for ends
        """
        raise NotImplementedError("Please implement this method for each subclass.")

    def plot_example_image(self, arr):
        """Helper method to take a multidimensional array or Tensor from a dataloader and plot the image embedded
        within. Implemented by each subclass because the dimensions of the data differ.

        Parameters
        ----------
        arr : torch.Tensor : array containing image to plot
        """
        raise NotImplementedError("Please implement this method for each subclass.")
