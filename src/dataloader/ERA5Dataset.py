import numpy as np
import matplotlib.pyplot as plt
from src.dataloader.WeatherDataset import WeatherDataset


class ERA5Dataset(WeatherDataset):

    def get_pairs(self, files, history=1):
        assert(self.file_naming_convention == "ERA5_npy")
        n = len(files)

        inps = []
        ends = files[history:]

        for i in range(n - history):
            inps.append(files[i:i + history])

        return np.array(inps), np.array(ends)

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
