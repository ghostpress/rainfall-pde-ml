import datetime
import matplotlib.pyplot as plt
from src.dataloader import WeatherDataLoader


class ERA5Loader(WeatherDataLoader.WeatherDataLoader):

    def get_date_from_filename(self, fname):
        """Helper method to take a filename containing a date string and return the date as a datetime.datetime object.
        Implemented by each subclass because the file naming convention used for each dataset is different.

        Naming convention for SST data: vv_fullDay_YYYYMMDD_[masked/filled].npy

        Parameters
        ----------
        fname : str : filename

        Returns
        -------
        date : datetime.datetime : date represented by the string in the filename
        """
        assert(self.file_naming_convention == "ERA5_npy")

        date = datetime.datetime.strptime(fname[11:19], "%Y%m%d").date()
        return date

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
