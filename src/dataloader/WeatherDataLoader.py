import matplotlib.pyplot as plt
import numpy as np
import datetime
import torch
import os
import math
# TODO: reorganize imports


class SatelliteDataLoader:

    def __init__(self, topdir, naming_conv, data_split, batchsize, timebase, hist=1, shuffle=True):
        self.parent_dir = topdir
        self.file_naming_convention = naming_conv
        self.split = data_split
        self.batchsize = batchsize
        self.timebase = timebase  # variable to determine whether each file represents one day, one hour, etc.
        self.history = hist
        self.shuffle = shuffle

        # Fields to create and save later
        self.train_files = None
        self.val_files = None
        self.test_files = None

        self.train_val_test_split_files()

        self.TrainLoader = None
        self.ValLoader = None
        self.TestLoader = None

    def get_date_from_filename(self, file):
        raise NotImplementedError("Please implement this method.")

    def get_total_time_passed(self, files):
        raise NotImplementedError("Please implement this method.")
        pass

    def train_val_test_cutoffs(self):
        """Helper method to create lists of filenames for the train, val, and test data splits. Uses the dates in the
        filenames to determine file order and split cutoffs.

        Returns
        -------
        cutoffs : list : date cutoffs for each split
        """

        all_files = os.listdir(self.path)
        all_files.sort()
        nfiles = len(all_files)
        ntimes = self.get_total_time_passed(all_files)

        start_date = self.get_date_from_filename(all_files[0])
        end_date = self.get_date_from_filename(all_files[nfiles - 1])

        cutoffs = []

        for i in range(3):
            delta = math.floor(ntimes * self.split[i])
            end = start_date + datetime.timedelta(days=delta)  # TODO: test that this works with ERA5 data
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

        all_files = os.listdir(self.path)
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
        return np.load(self.path + "/" + filename)

    def load_data_from_files(self, files):
        """Method to load data from a list of .npy files."""
        data = []
        for f in files:
            data.append(self.load_data_from_file(f))

        return np.array(data)

    def get_pairs(self, files):
        # TODO: basic version assumes files represent one region; inherited version customized to region and uses helper method
        pass

    def create_dataloader(self, files, dtype=torch.FloatTensor):
        # TODO: basic version assumes files represent one region; inherited version customized to region and uses helper method
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
                                             batch_size=self.batchsize, shuffle=self.shuffle)

        return loader

    def create_training(self):
        training_loader = self.create_dataloader(self.train_files)
        self.TrainLoader = training_loader

        first_, next_ = next(iter(training_loader))

        print("Training data: \n %d batches of size %d \n %d images" % (len(training_loader), self.batchsize,
                                                                        len(self.train_files)))
        print("Data shape: \n X: " + str(first_.shape) + ", y: " + str(next_.shape))

        print("Example image:")
        plt.imshow(first_[0][0])
        plt.show()

        return

    def create_validation(self):
        validation_loader = self.create_dataloader(self.val_files)
        self.ValLoader = validation_loader

        first_, next_ = next(iter(validation_loader))

        print("Validation data: \n %d batches of size %d \n %d images" % (len(validation_loader), self.batchsize,
                                                                          len(self.val_files)))
        print("Data shape: \n X: " + str(first_.shape) + ", y: " + str(next_.shape))

        print("Example image:")
        plt.imshow(first_[0][0])
        plt.show()

        return

    def create_test(self):
        test_loader = self.create_dataloader(self.test_files)
        self.TestLoader = test_loader

        first_, next_ = next(iter(test_loader))

        print("Test data: \n %d batches of size %d \n %d images" % (len(test_loader), self.batchsize,
                                                                    len(self.test_files)))
        print("Data shape: \n X: " + str(first_.shape) + ", y: " + str(next_.shape))

        print("Example image:")
        plt.imshow(first_[0][0])
        plt.show()

        return
