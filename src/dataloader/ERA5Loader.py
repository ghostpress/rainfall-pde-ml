import datetime
import matplotlib.pyplot as plt
from src.dataloader import WeatherDataLoader


class ERA5Loader(WeatherDataLoader.WeatherDataLoader):

    def get_date_from_filename(self, fname):
        assert(self.file_naming_convention == "ERA5_npy")

        date = datetime.datetime.strptime(fname[11:19], "%Y%m%d").date()
        return date

    def plot_example_image(self, arr):
        print("Example image:")
        plt.imshow(arr[0][0][0])
        plt.show()
