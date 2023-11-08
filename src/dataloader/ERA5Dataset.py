import numpy as np
import matplotlib.pyplot as plt
from src.dataloader.WeatherDataset import WeatherDataset


class ERA5Dataset(WeatherDataset):

    def train_val_test_split(self, split):
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
                                      variable_ids=self.variable_ids)
        ValidationDataset = ERA5Dataset(naming_conv=self.file_naming_convention, variable_files=val_files,
                                        variable_ids=self.variable_ids)
        TestingDataset = ERA5Dataset(naming_conv=self.file_naming_convention, variable_files=test_files,
                                     variable_ids=self.variable_ids)

        return TrainingDataset, ValidationDataset, TestingDataset

    def get_pairs(self, variable, use_wind=False, history=1):

        if not use_wind:
            files = self.variable_files[variable]
            assert(self.file_naming_convention == "ERA5_npy")
            n = len(files)

            inps = []
            ends = files[history:]

            for i in range(n - history):
                inps.append(files[i:i + history])

            return np.array(inps), np.array(ends)
        else:
            var_files = self.variable_files[variable]
            wind_files = self.variable_files["wind"]
            assert (self.file_naming_convention == "ERA5_npy")
            n = len(var_files)

            inps = []
            ends = var_files[history:]
            wind = []

            for i in range(n - history):
                inps.append(var_files[i:i + history])
                wind.append(wind_files[i])

            return np.array(inps), np.array(ends), np.array(wind)

    def plot_example_image(self, arr):
        """Helper method to take a multidimensional array or Tensor from a dataloader and plot the image embedded
        within. Implemented by each subclass because the dimensions of the data differ.

        Parameters
        ----------
        arr : torch.Tensor : array containing image to plot
        """
        print("Example image:")
        plt.imshow(arr[0][0][0])
        plt.show()
