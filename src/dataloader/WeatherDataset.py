import os
import numpy as np
import datetime
import math
from torch.utils.data import Dataset


class WeatherDataset(Dataset):

    def __init__(self, naming_conv, variable_ids, variable_files):
        assert naming_conv in ["ERA5_npy", "NEMO_npy"], f"Unsupported naming convention: {naming_conv}"
        self.file_naming_convention = naming_conv
        self.variable_ids = variable_ids
        self.variable_files = variable_files

        # Placeholders - create later
        #self.TrainingDataset = None
        #self.ValidationDataset = None
        #self.TestingDataset = None

    #def __init__(self, parent_path, naming_conv, variable_to_predict, other_variables):
    #    assert naming_conv in ["ERA5_npy", "NEMO_npy"], f"Unsupported naming convention: {naming_conv}"
    #    self.file_naming_convention = naming_conv
    #    self.parent_path = parent_path

        # Create dictionary of variables used; 0: variable to predict, 1,...,m: other variables
    #    self.variable_ids = dict()
    #    self.variable_ids[0] = variable_to_predict
    #    for i in range(len(other_variables)):
    #        self.variable_ids[i + 1] = other_variables[i]

        # Create dictionary of full paths to files for each variable
    #    self.variable_files = dict()
    #    for vname in self.variable_ids.values():
    #        path_to = self.parent_path + "/" + vname + "/"
    #        fnames = os.listdir(path_to)
    #        fnames.sort()
    #        self.variable_files[vname] = self._prepend_to_strings(fnames, path_to)

        # FIXME: change to new Dataset
    #    self.train_files = dict()
    #    self.val_files = dict()
    #    self.test_files = dict()

        #self.variable_files["primary"] = os.listdir(self.parent_path + "/" + variable_to_predict + "/")
        #self.path = parent_path

        #assert naming_conv in ["ERA5_npy", "NEMO_npy"], f"Unsupported naming convention: {naming_conv}"
        #self.file_naming_convention = naming_conv

        # load .npy files for primary variable (variable to predict)
        #primary_var_fnames = os.listdir(self.path + "/" + variable_to_predict)
        #self.variable = self.load_data_from_files(self.path + "/" + variable_to_predict + "/", primary_var_fnames)

        # load .npy files for each secondary variable (variables used to aid prediction but not directly predicted)
        #if len(other_variables) > 0:
        #    self.other_variables = dict()
        #    for vname in other_variables:
        #        secondary_var_fnames = os.listdir(self.path + "/" + vname)
        #        self.other_variables[vname] = self.load_data_from_files(self.path + "/" + vname + "/",
        #                                                                secondary_var_fnames)
        #self.train_files = []
        #self.val_files = []
        #self.test_files = []

    def __len__(self):
        #return len(self.variable)  # assumes the primary & secondary variables all have the same length
        var = self.variable_ids[0]
        return len(self.variable_files[var])

    def __getitem__(self, idx):
        #if len(self.other_variables.items()) == 0:
        #    return self.variable[idx]

        #elif len(self.other_variables.items()) == 1:
        #    return self.variable[idx], list(self.other_variables.values())[0][idx]

        #else:
        #    items = [self.variable[idx]]

        #    for name, var in self.other_variables.items():
        #        items.append({name: var})
        #    return items

        # Case 1: dataset contains just the primary variable
        if len(self.variable_ids) == 1:
            #print("case 1")
            var = self.variable_ids[0]
            #print(var)
            #print(self.variable_files[var])
            return self._load_data_from_file(self.variable_files[var][idx])#[idx]

        # Case 2: dataset contains the primary variable and one other variable
        elif len(self.variable_ids) == 2:
            #print("case 2")
            var_0 = self.variable_ids[0]
            var_1 = self.variable_ids[1]

            item1 = self._load_data_from_file(self.variable_files[var_0][idx])
            item2 = self._load_data_from_file(self.variable_files[var_1][idx])

            return item1, item2

        # Case 3: dataset contains the primary variable and many other variables
        else:
            #print("case 3")
            var_0 = self.variable_ids[0]
            items = [self._load_data_from_file(self.variable_files[var_0][idx])]#[idx]]

            for i in range(len(self.variable_files)):
                var_i = self.variable_ids[i+1]
                items.append(self._load_data_from_file(self.variable_files[var_i][idx]))#[idx])

            return items

    def _prepend_to_strings(self, strlist, pre):
        prep = strlist.copy()
        for i in range(len(prep)):
            prep[i] = pre + prep[i]

        return prep

    def _get_date_from_filename(self, fname):
        if self.file_naming_convention == "ERA5_npy":
            return datetime.datetime.strptime(fname.split("_")[-2], "%Y%m%d").date()
        else:
            return datetime.datetime.strptime(fname[8:16], "%Y%m%d").date()

    def _all_days(self, files):
        files.sort()

        first = self._get_date_from_filename(files[0])
        last = self._get_date_from_filename(files[len(files) - 1])

        delta = int((last - first) / datetime.timedelta(days=1))

        return delta

    # helper method
    def _train_val_test_cutoffs(self, split, variable):
        all_files = self.variable_files[variable] #os.listdir(self.parent_path + "/" + variable)
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
        raise NotImplementedError("Please implement this method for each subclass.")

    def plot_example_image(self, arr):
        raise NotImplementedError("Please implement this method for each subclass.")
