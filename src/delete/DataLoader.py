import datetime
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch

class DataLoader():

    def __init__(self, path, split, batchsize, history, shuffle=True):

        # Fields created and saved upon class initialization
        self.path = path
        self.split = split
        self.batchsize = batchsize
        self.history = history
        self.shuffle = shuffle

        # Fields to create and save later
        self.train_files = None
        self.val_files = None
        self.test_files = None

        self.train_val_test_split_files()

        self.TrainLoader = None
        self.ValLoader = None
        self.TestLoader = None

    def all_days(self, files):
        """Helper method to take a list of filenames and return the total number
           of days represented by the files in the list. 
    
           This method assumes a particular file naming convention that has been used 
           for this project. 
    
           File naming convention: sst_geo_yyyymmdd.nc_region_XX.npy
    
           Returns
           -------
           delta : int : the number of days represented by the inputted files
        """
    
        files.sort()
    
        first = datetime.datetime.strptime(files[0][8:16], "%Y%m%d").date()
        last = datetime.datetime.strptime(files[len(files)-1][8:16], "%Y%m%d").date()
    
        delta = int((last - first) / datetime.timedelta(days=1))
    
        return delta

    def train_val_test_cutoffs(self):
        """Helper method to create lists of filenames for the train, val, and test
           data splits. Uses the dates in the filenames to determine file order and 
           split cutoffs.
    
           This method assumes a particular file naming convention that has been used 
           for this project. 
    
           File naming convention: sst_geo_yyyymmdd.nc_region_XX.npy
            
           Returns
           -------
           cutoffs : list : date cutoffs for each split
        """
            
        all_files = os.listdir(self.path)
        all_files.sort()
        nfiles = len(all_files)
        ndays = self.all_days(all_files)
    
        start_date = datetime.datetime.strptime(all_files[0][8:16], "%Y%m%d").date()
        end_date = datetime.datetime.strptime(all_files[nfiles-1][8:16], "%Y%m%d").date()
    
        cutoffs = []
    
        for i in range(3):
            delta = math.floor(ndays*self.split[i])
            end = start_date + datetime.timedelta(days=delta)
            cutoffs.append(end)
        
            start_date = end
            
        # Because of rounding, some files may have been missed
        # Add these to the test split
        if cutoffs[2] < end_date:
            cutoffs[2] = end_date
    
        return cutoffs

    def train_val_test_split_files(self):
        """Method to split the data in a directory into training, validation, and
           test sets. Uses the helper method train_val_test_cutoffs().
            
           This method assumes a particular file naming convention that has been used 
           for this project. 
    
           File naming convention: sst_geo_yyyymmdd.nc_region_XX.npy
        """
    
        assert len(self.split) == 3, "Please include a % split for train, validation, and test sets."
    
        cutoffs = self.train_val_test_cutoffs()
    
        all_files = os.listdir(self.path)
        all_files.sort()
    
        train, val, test = [], [], []
    
        for f in all_files:
            file_date = datetime.datetime.strptime(f[8:16], "%Y%m%d").date()
        
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

    # Note: this method unique to SSTLoader
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

    def get_pairs_by_region(self, files, region):
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
    
        region_files = self.search_by_region(files, region)
    
        n = len(region_files)
    
        inps = []
        ends = region_files[self.history:]
    
        for i in range(n - self.history):
            inps.append(region_files[i:i+self.history])
    
        return np.array(inps), np.array(ends)

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

    def create_dataloader(self, files, dtype=torch.FloatTensor):
        """Method to create a PyTorch DataLoader object from a list of files and
        additional parameters.
    
        Parameters
        ----------
        files : str : a list of files holding data to put into the DataLoader
        dtype : torch.dtype : the data type for the DataLoader
    
        Returns
        -------
        loader : torch.utils.data.DataLoader : the DataLoader object
        """
        regions = self.all_regions(files)

        data = []
        ends = []

        for reg in regions:
            reg_pairs = self.get_pairs_by_region(files, reg) 
        
            for i in range(len(reg_pairs[0])):
                dat = self.load_data_from_files(reg_pairs[0][i]) 
                end = self.load_data_from_file(reg_pairs[1][i]) 
        
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

        print("Training data: \n %d batches of size %d \n %d images" %(len(training_loader), self.batchsize, len(self.train_files)))
        print("Data shape: \n X: " + str(first_.shape) + ", y: " + str(next_.shape))

        print("Example image:")
        plt.imshow(first_[0][0])
        plt.show()

        return

    def create_validation(self):
        validation_loader = self.create_dataloader(self.val_files)
        self.ValLoader = validation_loader

        first_, next_ = next(iter(validation_loader))

        print("Validation data: \n %d batches of size %d \n %d images" % (len(validation_loader), self.batchsize, len(self.val_files)))
        print("Data shape: \n X: " + str(first_.shape) + ", y: " + str(next_.shape))

        print("Example image:")
        plt.imshow(first_[0][0])
        plt.show()

        return

    def create_test(self):
        test_loader = self.create_dataloader(self.test_files)
        self.TestLoader = test_loader

        first_, next_ = next(iter(test_loader))

        print("Test data: \n %d batches of size %d \n %d images" % (len(test_loader), self.batchsize, len(self.test_files)))
        print("Data shape: \n X: " + str(first_.shape) + ", y: " + str(next_.shape))

        print("Example image:")
        plt.imshow(first_[0][0])
        plt.show()

        return
