import cdsapi

c = cdsapi.Client()

c.retrieve('reanalysis-era5-pressure-levels',
           {'product_type': 'reanalysis',
            'format': 'netcdf',
            'area': [11.6, -3.8, 4.3, 1.8],
            'variable': ['relative_humidity'],  # , 'specific_humidity', 'temperature'],
            'pressure_level': ['300', '500', '850'],
            'year': ['1990'],  # , '1991', '1992', '1993', '1994', '1995'],
            'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
            'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                    '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            },
           'test_download.nc')
