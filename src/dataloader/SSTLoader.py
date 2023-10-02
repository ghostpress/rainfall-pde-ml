import datetime

import src.dataloader


class SSTLoader(src.dataloader.WeatherDataLoader):

    def __init__(self):
        super().__init__()
        pass

    def get_date_from_filename(self, fname):
        assert(self.file_naming_convention == "NEMO_npy")

        date = datetime.datetime.strptime(fname[8:16], "%Y%m%d").date()
        return date

    def get_total_time_passed(self, files):
        """Helper method to take a list of filenames and return the total number of days represented by the files in the
        list. This method assumes a particular file naming convention that has been used for this project.

        File naming convention: sst_geo_yyyymmdd.nc_region_XX.npy

        Parameters
        ----------
        files : list : the list of filenames to scan

        Returns
        -------
        delta : int : the number of days represented by the inputted files
        """

        assert (self.file_naming_convention == "NEMO_npy")

        files.sort()

        first = self.get_date_from_filename(files[0][8:16])
        last = self.get_date_from_filename(files[len(files) - 1])

        delta = int((last - first) / datetime.timedelta(days=1))

        return delta

    def train_val_test_cutoffs(self):
        return super().train_val_test_cutoffs()



