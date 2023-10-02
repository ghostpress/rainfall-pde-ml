import datetime

import src.dataloader


class ERA5Loader(src.dataloader.WeatherDataLoader):

    def __init__(self):
        super().__init__()
        pass

    def get_date_from_filename(self, fname):
        assert(self.file_naming_convention == "NEMO_npy")

        date = datetime.datetime.strptime(fname[10:18], "%Y%m%d").date()  # TODO: test
        return date
    