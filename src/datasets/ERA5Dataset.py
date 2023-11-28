import numpy as np
import matplotlib.pyplot as plt
from src.datasets.WeatherDataset import WeatherDataset


class ERA5Dataset(WeatherDataset):

    def __init__(self, naming_conv, variable_ids, variable_files, history, hour=None):
        super().__init__(naming_conv, variable_ids, variable_files, history, hour)

    def train_val_test_split(self, split):
        """Method to split the data in a directory into training, validation, and test sets and return a new Dataset
        object. Uses the helper method train_val_test_cutoffs(). Returns a list of Dataset objects, one for each split.
        Parameters
        ----------
        split : list : list of fractions for each split
        """
        assert len(split) == 3, "Please include a % split for train, validation, and test sets."

        train_files = dict()
        val_files = dict()
        test_files = dict()

        for i in range(len(self.variable_ids)):
            print(f"Splitting dataset for variable {i}: {self.variable_ids[i]}")
            vname = self.variable_ids[i]
            cutoffs = self._train_val_test_cutoffs(split, vname)

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

        TrainingDataset = ERA5Dataset(naming_conv=self.file_naming_convention, variable_files=train_files,
                                      variable_ids=self.variable_ids, history=self.history, hour=self.hour)
        ValidationDataset = ERA5Dataset(naming_conv=self.file_naming_convention, variable_files=val_files,
                                        variable_ids=self.variable_ids, history=self.history, hour=self.hour)
        TestingDataset = ERA5Dataset(naming_conv=self.file_naming_convention, variable_files=test_files,
                                     variable_ids=self.variable_ids, history=self.history, hour=self.hour)

        return TrainingDataset, ValidationDataset, TestingDataset

    def get_pairs(self, variable, use_wind=True):
        """Method to separate data into inputs (X) and ends (y) for a given variable.
        For example: to use 4 previous days (X, hist=4) to predict the next day (y). The "pairs" are pairs of (X,y)
        inputs and ends.

        Parameters
        ----------
        variable : str : the variable to create pairs for
        use_wind : bool : whether to also return the wind arrays in the pairs

        Returns
        -------
        inps : np.array : filenames for inputs
        ends : np.array : filenames for ends
        wind : np.array : filenames for wind, if used
        """

        if not use_wind:
            files = self.variable_files[variable]
            assert(self.file_naming_convention == "ERA5_npy")
            n = len(files)

            inps = []
            ends = files[self.history:]

            for i in range(n - self.history):
                inps.append(files[i:i + self.history])

            return np.array(inps), np.array(ends)
        else:
            var_files = self.variable_files[variable]
            wind_files = self.variable_files["wind"]
            assert (self.file_naming_convention == "ERA5_npy")
            n = len(var_files)

            inps = []
            ends = var_files[self.history:]
            wind = []

            for i in range(n - self.history):
                inps.append(var_files[i:i + self.history])
                wind.append(wind_files[i])

            print("inps", np.array(inps).shape)
            print("ends", np.array(ends).shape)
            return np.array(inps), np.array(ends), np.array(wind)

    def plot_example_image(self, arr):
        """Helper method to take a multidimensional array or Tensor from a datasets and plot the image embedded
        within. Implemented by each subclass because the dimensions of the data differ.

        Parameters
        ----------
        arr : torch.Tensor : array containing image to plot
        """
        print("Example image:")
        plt.imshow(arr[0][0][0])
        plt.show()
