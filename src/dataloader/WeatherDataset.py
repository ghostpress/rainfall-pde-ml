import os
import numpy as np
import datetime
import math
from torch.utils.data import Dataset


class WeatherDataset(Dataset):

    def __init__(self, parent_path, naming_conv, variable_to_predict, other_variables):
        self.path = parent_path

        assert(naming_conv in ["ERA5_npy", "NEMO_npy"], f"Unsupported naming convention: {naming_conv}")
        self.file_naming_convention = naming_conv

        # load .npy files for primary variable (variable to predict)
        primary_var_fnames = os.listdir(self.path + "/" + variable_to_predict)
        self.variable = self.load_data_from_files(primary_var_fnames)

        # load .npy files for each secondary variable (variables used to aid prediction but not directly predicted)
        if len(other_variables) > 0:
            self.other_variables = dict()
            for vname in other_variables:
                secondary_var_fnames = os.listdir(self.path + "/" + vname)
                self.other_variables[vname] = self.load_data_from_files(secondary_var_fnames)

        self.train_files = []
        self.val_files = []
        self.test_files = []

    def __len__(self):
        return len(self.variable)  # assumes the primary & secondary variables all have the same length

    def __getitem__(self, idx):

        if len(self.other_variables.items()) == 0:
            return self.variable[idx]

        elif len(self.other_variables.items()) == 1:
            return self.variable[idx], list(self.other_variables.values())[0][idx]

        else:
            items = [self.variable[idx]]

            for name, var in self.other_variables.items():
                items.append({name: var})
            return items

    def _get_date_from_filename(self, fname):
        if self.file_naming_convention == "ERA5_npy":
            return datetime.datetime.strptime(fname[11:19], "%Y%m%d").date()
        else:
            return datetime.datetime.strptime(fname[8:16], "%Y%m%d").date()

    def _all_days(self, files):
        files.sort()

        first = self._get_date_from_filename(files[0])
        last = self._get_date_from_filename(files[len(files) - 1])

        delta = int((last - first) / datetime.timedelta(days=1))

        return delta

    # helper method
    # TODO: make sure the split happens for each variable - modify self.path? pass path to variable?
    def _train_val_test_cutoffs(self, split):
        all_files = os.listdir(self.path)
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

    # helper method
    # TODO: make sure the split happens for each variable - modify self.path? pass path to variable?
    def train_val_test_split(self, split):
        assert len(split) == 3, "Please include a % split for train, validation, and test sets."

        cutoffs = self._train_val_test_cutoffs(split)

        all_files = os.listdir(self.path)
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

        self.train_files = train
        self.val_files = val
        self.test_files = test
        return

    # helper method
    def _load_data_from_file(self, filename):
        """Helper method to load a single .npy file."""
        return np.load(self.path + "/" + filename)

    def load_data_from_files(self, files):
        """Method to load data from a list of .npy files."""
        data = []
        for f in files:
            data.append(self._load_data_from_file(f))

        return np.array(data)

    def get_pairs(self, files, history=1):
        raise NotImplementedError("Please implement this method for each subclass.")

    def plot_example_image(self):
        raise NotImplementedError("Please implement this method for each subclass.")

