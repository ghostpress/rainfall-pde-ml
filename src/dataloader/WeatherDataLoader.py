import datetime
import math
import numpy as np
import os
import torch


class WeatherDataLoader:

    def __init__(self, topdir, naming_conv, data_split, batchsize, hist=1):
        self.parent_dir = topdir
        self.file_naming_convention = naming_conv
        self.split = data_split
        self.batchsize = batchsize
        self.history = hist

        # Fields to create and save later
        self.train_files = None
        self.val_files = None
        self.test_files = None

        self.train_val_test_split_files()

        self.TrainLoader = None
        self.ValLoader = None
        self.TestLoader = None

    def get_date_from_filename(self, file):
        """Helper method to take a filename containing a date string and return the date as a datetime.datetime object.
        Implemented by each subclass because the file naming convention used for each dataset is different.

        Parameters
        ----------
        file : str : filename
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

    def all_days(self, files):
        """Helper method to take a list of filenames and return the total number of days represented by the files in
        the list.

        Parameters
        ----------
        files : list : list of filenames
        Returns
        -------
        delta : int : the number of days represented by the inputted files
        """
        files.sort()

        first = self.get_date_from_filename(files[0])
        last = self.get_date_from_filename(files[len(files) - 1])

        delta = int((last - first) / datetime.timedelta(days=1))

        return delta
        pass

    def train_val_test_cutoffs(self):
        """Helper method to create lists of filenames for the train, val, and test data splits. Uses the dates in the
        filenames to determine file order and split cutoffs.

        Returns
        -------
        cutoffs : list : date cutoffs for each split
        """

        all_files = os.listdir(self.parent_dir)
        all_files.sort()
        nfiles = len(all_files)
        ndays = self.all_days(all_files)

        start_date = self.get_date_from_filename(all_files[0])
        end_date = self.get_date_from_filename(all_files[nfiles - 1])

        cutoffs = []

        for i in range(3):
            delta = math.floor(ndays * self.split[i])
            end = start_date + datetime.timedelta(days=delta)
            cutoffs.append(end)

            start_date = end

        # Because of rounding, some files may have been missed
        # Add these to the test split
        if cutoffs[2] < end_date:
            cutoffs[2] = end_date

        return cutoffs

    def train_val_test_split_files(self):
        """Method to split the data in a directory into training, validation, and test sets. Uses the helper method
        train_val_test_cutoffs().
        """
        assert len(self.split) == 3, "Please include a % split for train, validation, and test sets."

        cutoffs = self.train_val_test_cutoffs()

        all_files = os.listdir(self.parent_dir)
        all_files.sort()

        train, val, test = [], [], []

        for f in all_files:
            file_date = self.get_date_from_filename(f)

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

    def load_data_from_file(self, filename):
        """Helper method to load a single .npy file."""
        return np.load(self.parent_dir + "/" + filename)

    def load_data_from_files(self, files):
        """Method to load data from a list of .npy files."""
        data = []
        for f in files:
            data.append(self.load_data_from_file(f))

        return np.array(data)

    def get_pairs(self, files):
        """Method to separate data into inputs (X) and ends (y), for example to use 4 previous days (hist=4) to
        predict the next day. The "pairs" are pairs of (X,y) inputs and ends.

        Parameters
        ----------
        files : list : list of files from which to get pairs

        Returns
        -------
        inps : np.ndarray : filenames for inputs
        ends : list : filenames for ends
        """
        print("here, in Weather class")
        n = len(files)

        inps = []
        ends = files[self.history:]

        for i in range(n - self.history):
            inps.append(files[i:i + self.history])

        return np.array(inps), np.array(ends)

    def create_dataloader(self, files, dtype=torch.FloatTensor, shuffle=True):
        data = []
        ends = []

        pairs = self.get_pairs(files)

        for i in range(len(pairs[0])):
            dat = self.load_data_from_files(pairs[0][i])
            end = self.load_data_from_file(pairs[1][i])

            data.append(dat)
            ends.append(end)

        final_data = torch.from_numpy(np.array(data)).type(dtype)
        final_ends = torch.from_numpy(np.array(ends)).type(dtype)

        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(final_data, final_ends),
                                             batch_size=self.batchsize, shuffle=shuffle)

        return loader

    def create_training(self, shuffle=True):
        training_loader = self.create_dataloader(self.train_files, shuffle=shuffle)
        self.TrainLoader = training_loader

        first_, next_ = next(iter(training_loader))

        print("Training data: \n %d batches of size %d \n %d images" % (len(training_loader), self.batchsize,
                                                                        len(self.train_files)))
        print("Data shape: \n X: " + str(first_.shape) + ", y: " + str(next_.shape))
        self.plot_example_image(first_)

        return

    def create_validation(self, shuffle=True):
        validation_loader = self.create_dataloader(self.val_files, shuffle=shuffle)
        self.ValLoader = validation_loader

        first_, next_ = next(iter(validation_loader))

        print("Validation data: \n %d batches of size %d \n %d images" % (len(validation_loader), self.batchsize,
                                                                          len(self.val_files)))
        print("Data shape: \n X: " + str(first_.shape) + ", y: " + str(next_.shape))
        self.plot_example_image(first_)

        return

    def create_test(self, shuffle=True):
        test_loader = self.create_dataloader(self.test_files, shuffle=shuffle)
        self.TestLoader = test_loader

        first_, next_ = next(iter(test_loader))

        print("Test data: \n %d batches of size %d \n %d images" % (len(test_loader), self.batchsize,
                                                                    len(self.test_files)))
        print("Data shape: \n X: " + str(first_.shape) + ", y: " + str(next_.shape))
        self.plot_example_image(first_)

        return
